"""
Validity

 The README defines it as the fraction of outputs that parse as JSON and validate against ExtractionResult, over all eval chunks.



 Your schema is strict: node types and relations are enums, and names must match the identifier regex. So the validator also catches
 invented relation types and multi-word names.

 A truncated JSON (generation stopped at max_new_tokens) fails to parse. The README says to count it as invalid, not skip it and not
 let it crash the run.

 Precision and recall

"""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from data.schema import ExtractionResult
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


def load_split(path: Path) -> list[tuple[Chunk, ExtractionResult]]:
    """
    read an eval split: one JSON object per line,

        {"chunk": {"file_path", "first", "last", "text", "source_type"},
         "messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

    the chunk is stored next to the chat record so the run can rebuild the
    exact prompt without the corpus on disk. the assistant
    message is the hand-corrected gold.
    it is parsed without a try/except:
    a gold record that doesn't validate is broken data and must stop the run.
    """
    pairs = []
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            chunk = Chunk(**record["chunk"])
            gold_result = ExtractionResult.model_validate_json(
                record["messages"][1]["content"]
            )
            pairs.append((chunk, gold_result))
    return pairs


def run(
    pairs: list[tuple[Chunk, ExtractionResult]],
    out_dir: Path,
    adapter: str | None = None,
    constrained: bool = True,
) -> dict[str, float]:
    """
    check if chunk is validated, success counts as valid.
    an exception counts as invalid.
    Validity rate = valid outputs ÷ all eval chunks.

    Then:
    - Precision = TP ÷ (TP + FP): of what the model said, how much was right.
    - Recall = TP ÷ (TP + FN): of what was there, how much the model found.

    raw outputs go to out_dir/outputs.jsonl (one line per chunk) and the
    aggregates to out_dir/metrics.json, so a run can be re-scored or
    inspected for failure modes without running the model again.
    """
    # if we dont have any chunks raise error
    if not pairs:
        raise ValueError("No chunks provided")

    # load model
    model, tokenizer = load_hf_model(adapter=adapter)
    generator = build_generator(model, tokenizer) if constrained else None
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = (out_dir / "outputs.jsonl").open("w")

    # check for validity
    valid = tp_total = fp_total = fn_total = 0
    for chunk, gold_result in pairs:
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

        outputs.write(
            json.dumps(
                {
                    "file_path": chunk.file_path,
                    "first": chunk.first,
                    "last": chunk.last,
                    "raw": raw,
                    "valid": result is not None,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
            + "\n"
        )
    outputs.close()

    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0

    # validity rate
    validity_rate = valid / len(pairs)
    metrics = {"validity_rate": validity_rate, "precision": precision, "recall": recall}

    run_info = {
        "adapter": adapter,
        "constrained": constrained,
        "chunks": len(pairs),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics | run_info, indent=2) + "\n"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score one setup on an eval split")
    parser.add_argument("--split", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument(
        "--out", type=Path, required=True, help="e.g. results/base_constrained"
    )
    parser.add_argument("--adapter", default=None, help="LoRA adapter id or path")
    parser.add_argument("--unconstrained", action="store_true")
    args = parser.parse_args()

    metrics = run(
        load_split(args.split),
        args.out,
        adapter=args.adapter,
        constrained=not args.unconstrained,
    )
    print(f"validity:  {metrics['validity_rate']:.2%}")
    print(f"precision: {metrics['precision']:.2%}")
    print(f"recall:    {metrics['recall']:.2%}")
