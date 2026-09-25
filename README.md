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

`src/generate.py` exposes two paths over the same prompts (`build_prompt` routes by `Chunk.source_type`):

| Function | Decoding | Returns |
| --- | --- | --- |
| `extract(generator, chunk)` | Outlines FSM over `ExtractionResult` | raw text |
| `unconstrained_extract(hf_model, hf_tokenizer, chunk)` | plain `model.generate` | raw text |

Both return raw text and leave parsing/validation to the evaluator. Constrained decoding only guarantees a schema-valid *prefix*: output truncated at `max_new_tokens` (default 512) is still unparseable JSON, and must be counted as invalid rather than raising or being dropped.

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

**Corpus:** [vLLM v0.10.1](https://github.com/vllm-project/vllm/tree/v0.10.1) source (2,874 files), unpacked to `data/raw/vllm-0.10.1/`. Not committed — it is third-party code and fetched on demand (see [Setup](#setup)); the chunk text that is actually used lives inside the committed JSONL splits.

## Setup

Python 3.10–3.13

```bash
uv sync
```

Fetch the corpus:

```bash
mkdir -p data/raw
curl -L https://github.com/vllm-project/vllm/archive/refs/tags/v0.10.1.tar.gz | tar -xz -C data/raw
```

Run from the repo root — `data.schema` resolves as a namespace package relative to it.

## Layout

```
data/schema.py          node/relation types + ExtractionResult
data/raw/               source corpus (gitignored, fetched in Setup)
src/chunking/           AST (Python) and header-based (markdown/text) chunkers
src/extraction/prompts/ code and text extraction prompts
src/generate.py         constrained + unconstrained inference
src/build_dataset.py    dataset construction (empty)
```
