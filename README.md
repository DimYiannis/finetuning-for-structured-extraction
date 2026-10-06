# Qwen3-0.6B LoRA fine-tune for schema-constrained triple extraction

Fine-tuning Qwen3-0.6B with LoRA to extract knowledge-graph triples that conform to a fixed schema.

Companion to [constrained-graphrag](https://github.com/DimYiannis/Constrained-GraphRag), which uses the same model for entity/relation extraction and enforces the schema at decode time with [Outlines](https://github.com/dottxt-ai/outlines).

> **Status:** work in progress. Chunking, prompts, schema and the inference wrapper are in place. The dataset, evaluator, training run and all results are **not yet produce.

---

## Question

constrained-graphrag guarantees schema-shaped output by masking logits with an FSM compiled from a Pydantic model. That fixes *syntax*, not *semantics*: the model can still pick the wrong node type, the wrong relation, or miss triples. This project asks:

1. Does fine-tuning alone buy schema conformance?
2. Does it improve triple accuracy over the constrained base model?
3. Do fine-tuning and constrained decoding compound, or overlap?

## Lineage

| Here | Upstream |
| --- | --- |
| `data/schema.py` | `src/extraction/schema.py` |
| `src/chunking/` | chunkers (AST for Python, header-based for markdown/text) |
| `src/extraction/prompts/` | code and text extraction prompts |
| `src/generate.py` → `extract` | `src/extraction/extractor.py` |

New in this repo: `unconstrained_extract` (upstream only ever ran constrained), and `extract` returns raw text instead of a validated object (see [Inference](#inference)).

## Schema

Defined once in `data/schema.py`.

**Node types:** `Function`, `Class`, `Module`, `Concept`, `Entity`, `Chunk`
**Relation types:** `CALLS`, `IMPORTS`, `INHERITS_FROM`, `DEFINED_IN`, `MENTIONED_IN`, `RELATES_TO`, `REFERENCES`

The model extracts a subset: `Chunk` nodes are created during chunking, and `MENTIONED_IN` edges are added by the graph loader for every entity extracted from a chunk.

Output shape per chunk (`ExtractionResult`):

```json
{
  "entities": [{"name": "resolve", "node_type": "Function"}],
  "relationships": [{"subject": "resolve", "relation": "CALLS", "target": "validate"}]
}
```

Entity names, subjects and targets must match `^[A-Za-z_][A-Za-z0-9_.]*$` — identifiers, not free text.

## Inference

`src/generate.py` loads the model once (`load_hf_model`, optionally with a LoRA `adapter`) and exposes two paths over it:

| Function | Decoding | Returns |
| --- | --- | --- |
| `extract(generator, hf_tokenizer, chunk)` | Outlines FSM over `ExtractionResult` | raw text |
| `unconstrained_extract(hf_model, hf_tokenizer, chunk)` | plain `model.generate` | raw text |

Both return raw text and leave parsing/validation to the evaluator. Constrained decoding only guarantees a schema-valid *prefix*: output truncated at `max_new_tokens` (default 512) is still unparseable JSON, and must be counted as invalid rather than raising or being dropped.

Every row of the comparison uses the same inference setup:

- **Chat template.** The extraction prompt (`build_prompt` routes by `Chunk.source_type`) is sent as a single user turn through Qwen3's chat template — the same format SFT trains on, so train and serve match. This differs from upstream constrained-graphrag, which feeds the prompt as a raw string; the "base + constrained" row is therefore the upstream approach *with* the chat template.
- **Thinking disabled.** `enable_thinking=False` puts an empty `<think></think>` block in the generation prompt, which is also what the template renders before the answer in non-reasoning training data. The model starts directly at the JSON.
- **Greedy decoding.** Qwen3's shipped sampling config is overridden (`do_sample=False`) so eval numbers are reproducible.
- **Device.** CUDA in bf16 where the GPU supports it natively, else fp16 (T4 only emulates bf16); MPS/CPU in fp32.

## Evaluation plan

Fixed held-out split (~15%), hand-corrected, reused unchanged for every run.

- **Schema validity rate** — fraction of outputs that parse as JSON *and* validate against `ExtractionResult`, over all eval chunks.
- **Triple accuracy** — precision/recall over `(subject, relation, target)` triples against the reference.
- **Mean output length** and failure modes (truncation, parse errors, invalid enum values, hallucinated relations, duplicates).

Raw outputs are saved under `results/`, not just aggregates.

## Dataset

*Not yet built.* Planned: 300–800 chunks from the constrained-graphrag corpus, candidate labels from a larger model, eval split hand-corrected in full, training split spot-checked. Chat-formatted JSONL:

```json
{"messages": [
  {"role": "user", "content": "<extraction prompt + chunk>"},
  {"role": "assistant", "content": "<ExtractionResult JSON>"}
]}
```

The user turn is the exact inference-time prompt from `src/extraction/prompts/`.

**Corpus:** the same corpus used by constrained-graphrag — [`data/raw/vllm-0.10.1`](https://github.com/DimYiannis/Constrained-GraphRag/tree/47a33e0fdb1790736c18333771c951679e837f8c/data/raw/vllm-0.10.1).

## Setup

Python 3.10–3.13

```bash
uv sync
```

Run from the repo root — `data.schema` resolves as a namespace package relative to it.

## Layout

```
data/schema.py          node/relation types + ExtractionResult
data/raw/               source corpus (gitignored; from constrained-graphrag, see Corpus)
src/chunking/           AST (Python) and header-based (markdown/text) chunkers
src/extraction/prompts/ code and text extraction prompts
src/generate.py         constrained + unconstrained inference
src/build_dataset.py    dataset construction (empty)
```

## Sources

For the evaluator (`src/evaluate.py`):

| Topic | Link |
| --- | --- |
| Parse and validate JSON with Pydantic (`model_validate_json`) | [Pydantic: JSON](https://docs.pydantic.dev/latest/concepts/json/) |
| Pydantic models and validation | [Pydantic: Models](https://docs.pydantic.dev/latest/concepts/models/) |
| `ValidationError`, raised on invalid output | [Pydantic: ValidationError](https://docs.pydantic.dev/latest/api/pydantic_core/#pydantic_core.ValidationError) |
| Set operations for comparing triple sets | [Python: set types](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset) |
| Precision, recall and F-measure | [scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics) |
| Precision and recall, definitions | [Wikipedia: Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) |
| JSON parsing | [Python: `json`](https://docs.python.org/3/library/json.html) |

For inference (`src/generate.py`):

| Topic | Link |
| --- | --- |
| Chat templates, `apply_chat_template` | [Transformers: Chat templating](https://huggingface.co/docs/transformers/chat_templating) |
| Qwen3 model card (thinking mode, `enable_thinking`) | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| `generate()` and `GenerationConfig` (greedy override) | [Transformers: Text generation](https://huggingface.co/docs/transformers/main_classes/text_generation) |
| Greedy decoding vs sampling | [Transformers: Generation strategies](https://huggingface.co/docs/transformers/generation_strategies) |
| Outlines overview | [Outlines docs](https://dottxt-ai.github.io/outlines/latest/) |
| Wrapping a Hugging Face model (`from_transformers`) | [Outlines: Transformers models](https://dottxt-ai.github.io/outlines/latest/features/models/transformers/) |
| Constrained generation (`Generator`, `output_type`) | [Outlines: Generator](https://dottxt-ai.github.io/outlines/latest/features/core/generator/) |
| Outlines source and issues | [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines) |
| PEFT and LoRA overview | [PEFT docs](https://huggingface.co/docs/peft/index) |
| Loading an adapter (`PeftModel.from_pretrained`) | [PEFT: PeftModel](https://huggingface.co/docs/peft/package_reference/peft_model) |
| CUDA availability check | [PyTorch: `torch.cuda.is_available`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html) |
| bf16 support check (`including_emulation`) | [PyTorch: `torch.cuda.is_bf16_supported`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html) |
| Apple GPU (MPS) backend | [PyTorch: MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html) |
