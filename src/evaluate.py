"""
Validity

 The README defines it as the fraction of outputs that parse as JSON and validate against ExtractionResult, over all eval chunks.

 1. Take the raw text from extract or unconstrained_extract.
 2. Try to turn it into an ExtractionResult. The model_validate_json method on a Pydantic model does both the parse and the schema
    check in one call. Per the Pydantic JSON docs, it raises ValidationError on bad input.
 3. Success counts as valid. An exception counts as invalid.
 4. Validity rate = valid outputs ÷ all eval chunks.

 Your schema is strict: node types and relations are enums, and names must match the identifier regex. So the validator also catches
 invented relation types and multi-word names.

 A truncated JSON (generation stopped at max_new_tokens) fails to parse. The README says to count it as invalid, not skip it and not
 let it crash the run.

 Precision and recall

 1. For each valid output, build a set of triples: {(subject, relation, target), ...}. Do the same for the hand-corrected reference.
 2. Compare them:
   - TP (true positives) = triples in both sets, pred & gold
   - FP (false positives) = predicted but not in the reference, pred - gold
   - FN (false negatives) = in the reference but missed, gold - pred
 3. Then:
   - Precision = TP ÷ (TP + FP): of what the model said, how much was right.
   - Recall = TP ÷ (TP + FN): of what was there, how much the model found.
"""

from pathlib import Path

from pydantic import ValidationError

from data.schema import ExtractionResult
from src.chunking.ast_chunker import chunk_python
from src.chunking.spans import Chunk
from src.generate import (
    build_generator,
    extract,
    load_hf_model,
    unconstrained_extract,
)


def validate(raw: str) -> ExtractionResult | None:
    try:
        return ExtractionResult.model_validate_json(raw)
    except ValidationError as error:
        print(error)
        return None


def run(
    chunks: list[Chunk], adapter: str | None = None, constrained: bool = True
) -> None:
    model, tokenizer = load_hf_model(adapter=adapter)
    generator = build_generator(model, tokenizer) if constrained else None

    for chunk in chunks:
        if constrained:
            raw = extract(generator, tokenizer, chunk)
        else:
            raw = unconstrained_extract(model, tokenizer, chunk)
        result = validate(raw)
        print(f"{chunk.file_path} [{chunk.first}:{chunk.last}] valid={result is not None}")


if __name__ == "__main__":
    path = "data/raw/vllm-0.10.1/vllm/lora/request.py"
    chunks = chunk_python(Path(path).read_text(), path)

    run(chunks)
