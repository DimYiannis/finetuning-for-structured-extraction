"""
runs Qwen3-0.6B over a chunk to extract entities/nodes-relationships/edges,
constrained (Outlines) or unconstrained (plain HF .generate).

Both paths return raw decoded text; parsing/validation is left to
evaluate.py. Outlines' FSM only guarantees every sampled token keeps the
output a valid *prefix* of the schema - if generation hits max_new_tokens
the JSON is cut off and still fails to parse, so the constrained path
can't be trusted to validate here either.
"""

import outlines
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.chunking.spans import Chunk
from src.extraction.prompts import build_prompt
from data.schema import ExtractionResult

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS = 512


def _device_and_dtype() -> tuple[str, torch.dtype]:
    """
    pick where the model run

    check if hardware suports bf16 else switch to fp16
    MPS/CPU stay on f32 format.

    a 0.6b model fits either way and fp32 avoids precision suprises

    including_emulation=False: the default (True) counts software-emulated
    bf16, which a T4 would report as supported and wrongly pick over fp16
    """
    if torch.cuda.is_avalaible():
        bf16_native = torch.cuda.os_bf16_supported(including_emulation=False)
        dtype = torch.bfloat16 if bf16_native else torch.float16
        return "cuda", dtype
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_hf_model(model: str = DEFAULT_MODEL_NAME, adapter: str | None = None):
    """
    load the raw Hugging Face model + tokenizer, once.

    args:
        hf model
        adapter: optional LoRA adapter id or path, applied on top of the
        base for the fine-tuned rows.

    shared by both the constrained path (load_model wraps this in
    Outlines) and the unconstrained path.
    """
    device, dtype = _device_and_dtype()
    hf_model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
    if adapter is not None:
        hf_model = PeftModel.from_pretrained(hf_model, adapter)
    hf_model.to(device).eval()

    # Qwen3 ships a sampling generation_config (temperature/top_p/top_k);
    # override it on the model so both paths (Outlines calls .generate
    # with the model's config too) decode greedily.
    generation_config = hf_model.generation_config
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None

    hf_tokenizer = AutoTokenizer.from_pretrained(model)
    return hf_model, hf_tokenizer


def build_generator(hf_model, hf_tokenizer):
    """
    build a reusable constrained generator for ExtractionResults

    wraps the loaded model, so both paths run on the same weights.
    build once, called per chunk, avoids recompiling the FSM
    constraint for every one of all the chunks.
    """
    model = outlines.from_transformers(hf_model, hf_tokenizer)
    return outlines.Generator(model, output_type=ExtractionResult)


def extract(
    generator, chunk: Chunk, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> str:
    """
    extract nodes/edges from one chunk

    args:
        generator: constrained generator for entities/relationships
        chunk: routed to the code/text prompt by chunk.source_type
        max_new_tokens

    return:
        raw decoded text - schema-conforming prefix, but may be truncated
        at max_new_tokens, so not guaranteed to parse.
    """
    prompt = build_prompt(chunk)
    return generator(prompt, max_new_tokens=max_new_tokens)


def unconstrained_extract(
    hf_model,
    hf_tokenizer,
    chunk: Chunk,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """
    extract nodes/edges from one chunk, unconstrained.

    args:
        hf_model, hf_tokenizer: loaded once via load_hf_model, reused
            across every chunk in a run
        chunk: routed to the code/text prompt by chunk.source_type
        max_new_tokens

    return:
        raw decoded text - no JSON/schema guarantee.
    """
    prompt = build_prompt(chunk)
    model_inputs = hf_tokenizer(prompt, return_tensors="pt")
    output_ids = hf_model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    new_tokens = output_ids[0][
        model_inputs["input_ids"].shape[1] :
    ]  # return only new generated tokens
    return hf_tokenizer.decode(new_tokens, skip_special_tokens=True)
