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
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.chunking.spans import Chunk
from src.extraction.prompts import build_prompt
from data.schema import ExtractionResult

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS = 512


def load_hf_model(model: str = DEFAULT_MODEL_NAME):
    """
    load the raw Hugging Face model + tokenizer, once.

    shared by both the constrained path (load_model wraps this in
    Outlines) and the unconstrained path.
    """
    hf_model = AutoModelForCausalLM.from_pretrained(model)
    hf_tokenizer = AutoTokenizer.from_pretrained(model)
    return hf_model, hf_tokenizer


def load_model(model: str = DEFAULT_MODEL_NAME):
    """
    wrap a Hugging face causal LM in an Outlines model
    """
    hf_model, hf_tokenizer = load_hf_model(model)
    return outlines.from_transformers(hf_model, hf_tokenizer)


def build_generator(model):
    """
    build a reusable constrained generator for ExtractionResults

    build once, called per chunk, avoids recompiling the FSM
    constraint for every one of all the chunks.
    """
    generator = outlines.Generator(model, output_type=ExtractionResult)
    return generator


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

