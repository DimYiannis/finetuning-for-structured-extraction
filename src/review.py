"""
hand-correction round trip for labelled splits.

    export: candidates JSONL -> one pair of files per record, easy to edit
            <dir>/NNN.chunk.txt   the chunk (read-only reference)
            <dir>/NNN.gold.json   the candidate labels, one item per line
    import: edited gold files + candidates JSONL -> final split JSONL,
            every gold validated against ExtractionResult, plus how much
            the hand pass changed (labeller precision/recall vs the
            corrected gold), which goes in the README data card.

    uv run python -m src.review export --candidates data/eval_candidates.jsonl --dir data/review/eval
    (edit the .gold.json files)
    uv run python -m src.review import --candidates data/eval_candidates.jsonl --dir data/review/eval --out data/eval.jsonl
"""

import argparse
import json
from pathlib import Path

from data.schema import ExtractionResult
from src.evaluate import score, to_triples


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _gold_text(result: ExtractionResult) -> str:
    """pretty JSON with one entity / relationship per line, so diffs and edits stay readable"""
    entities = [e.model_dump(mode="json") for e in result.entities]
    relationships = [r.model_dump(mode="json") for r in result.relationships]
    lines = ['{', '  "entities": [']
    lines += [f"    {json.dumps(e)}," for e in entities]
    if entities:
        lines[-1] = lines[-1].rstrip(",")
    lines += ["  ],", '  "relationships": [']
    lines += [f"    {json.dumps(r)}," for r in relationships]
    if relationships:
        lines[-1] = lines[-1].rstrip(",")
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"


def export(candidates: Path, out_dir: Path) -> None:
    records = _read_jsonl(candidates)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records):
        chunk = record["chunk"]
        gold_path = out_dir / f"{i:03d}.gold.json"
        if gold_path.exists():
            raise FileExistsError(f"{gold_path} exists - refusing to overwrite hand edits")
        header = f"# {chunk['file_path']} [{chunk['first']}:{chunk['last']}] ({chunk['source_type']})\n\n"
        (out_dir / f"{i:03d}.chunk.txt").write_text(header + chunk["text"])
        candidate = ExtractionResult.model_validate_json(record["messages"][1]["content"])
        gold_path.write_text(_gold_text(candidate))
    print(f"exported {len(records)} records to {out_dir}")


def import_(candidates: Path, in_dir: Path, out: Path) -> None:
    records = _read_jsonl(candidates)
    changed = tp_total = fp_total = fn_total = 0
    with out.open("w") as f:
        for i, record in enumerate(records):
            gold_path = in_dir / f"{i:03d}.gold.json"
            try:
                corrected = ExtractionResult.model_validate_json(gold_path.read_text())
            except Exception as error:
                raise SystemExit(f"{gold_path}: {error}") from None
            candidate = ExtractionResult.model_validate_json(record["messages"][1]["content"])
            # candidate labels scored against the hand-corrected gold
            tp, fp, fn = score(to_triples(candidate), to_triples(corrected))
            tp_total += tp
            fp_total += fp
            fn_total += fn
            changed += fp + fn > 0
            record["messages"][1]["content"] = corrected.model_dump_json()
            f.write(json.dumps(record) + "\n")

    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    print(f"wrote {len(records)} records to {out}")
    print(f"records with changed triples: {changed}/{len(records)}")
    print(f"triples removed: {fp_total}  added: {fn_total}  kept: {tp_total}")
    print(f"labeller vs corrected gold: precision {precision:.2%}  recall {recall:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hand-correction round trip")
    sub = parser.add_subparsers(dest="command", required=True)
    p_export = sub.add_parser("export")
    p_export.add_argument("--candidates", type=Path, required=True)
    p_export.add_argument("--dir", type=Path, required=True)
    p_import = sub.add_parser("import")
    p_import.add_argument("--candidates", type=Path, required=True)
    p_import.add_argument("--dir", type=Path, required=True)
    p_import.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "export":
        export(args.candidates, args.dir)
    else:
        import_(args.candidates, args.dir, args.out)
