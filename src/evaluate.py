"""
Validity

 The README defines it as the fraction of outputs that parse as JSON and validate against ExtractionResult, over all eval chunks.



 Your schema is strict: node types and relations are enums, and names must match the identifier regex. So the validator also catches
 invented relation types and multi-word names.

 A truncated JSON (generation stopped at max_new_tokens) fails to parse. The README says to count it as invalid, not skip it and not
 let it crash the run.

 Precision and recall

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
    """
    - Take the raw text from extract or unconstrained_extract.
    - Try to turn it into an ExtractionResult.
    (The model_validate_json method on a Pydantic model does both the parse and the schema
    check in one call. Per the Pydantic JSON docs, it raises ValidationError on bad input.)
    """
    try:
        return ExtractionResult.model_validate_json(raw)
    except ValidationError as error:
        print(error)
        return None


def to_triples(valid: ExtractionResult) -> set[tuple[str, str, str]]:
    """
    build a set of triples:
        {(subject, relation, target), ...} out of a valid output
        do same for the hand-corrected reference
    """
    return {(r.subject, r.relation, r.target) for r in valid.relationships}


def score(
    pred: set[tuple[str, str, str]], gold: set[tuple[str, str, str]]
) -> tuple[int, int, int]:
    """
    Compare:
    - TP (true positives) = triples in both sets, pred & gold
    - FP (false positives) = predicted but not in the reference, pred - gold
    - FN (false negatives) = in the reference but missed, gold - pred
    """
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return (tp, fp, fn)


def run(
    chunks: list[Chunk], adapter: str | None = None, constrained: bool = True
) -> dict[str, float]:
    """
    check if chunk is validated, success counts as valid.
    an exception counts as invalid.
    Validity rate = valid outputs ÷ all eval chunks.

    Then:
    - Precision = TP ÷ (TP + FP): of what the model said, how much was right.
    - Recall = TP ÷ (TP + FN): of what was there, how much the model found.

    """
    # golden example
    gold_result = ExtractionResult(
        entities=[],
        relationships=[
            {"subject": "resolve", "relation": "CALLS", "target": "validate"},
            {"subject": "LoRAResolver", "relation": "DEFINED_IN", "target": "resolver"},
        ],
    )

    # if we dont have any chunks raise error
    if not chunks:
        raise ValueError("No chunks provided")

    # load model
    model, tokenizer = load_hf_model(adapter=adapter)
    generator = build_generator(model, tokenizer) if constrained else None

    # check for validity
    valid = tp_total = fp_total = fn_total = 0
    for chunk in chunks:
        if constrained:
            raw = extract(generator, tokenizer, chunk)
        else:
            raw = unconstrained_extract(model, tokenizer, chunk)
        result = validate(raw)
        print(
            f"{chunk.file_path} [{chunk.first}:{chunk.last}] valid={result is not None}"
        )

        gold = to_triples(gold_result)
        if result is not None:
            pred = to_triples(result)
            valid += 1
        else:
            pred = set()

        tp, fp, fn = score(pred, gold)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0

    # validity rate
    validity_rate = valid / len(chunks)
    return {"validity_rate": validity_rate, "precision": precision, "recall": recall}


if __name__ == "__main__":
    path = "data/raw/vllm-0.10.1/vllm/lora/request.py"
    chunks = chunk_python(Path(path).read_text(), path)
    print(f"validity: {run(chunks):.2%}")
