"""
script that picks which chunks go into the dataset
"""

import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from src.chunking.chunk_corpus import chunk_corpus
from src.chunking.spans import Chunk
from src.extraction.prompts import build_prompt
from src.generate import build_chat_prompt

SEED = 0
EVAL_FRACTION = 0.15
MIN_CHARS = 100
MAX_PROMPT_TOKENS = 1700
MAX_PER_FILE = 2

# chunks per group; "json" is absent, so it is never sampled
EVAL_QUOTAS = {"python": 34, "docs": 8, "other": 8}  # 50
TRAIN_QUOTAS = {"python": 200, "docs": 50, "other": 50}  # 300


def group(chunk: Chunk) -> str:
    """
    sort chunks by file extension
    """
    suffix = Path(chunk.file_path).suffix.lower()
    if suffix == ".py":
        return "python"
    elif suffix in {".md", ".rst", ".txt"}:
        return "docs"
    elif suffix == ".json":
        return "json"
    else:
        return "other"


def keep(chunk: Chunk, tokenizer) -> bool:
    """
    determine whether a chunk
    is worth putting in the dataset
    """
    # check if its only comments
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    only_comments = all(line.startswith("#") for line in lines)

    if group(chunk) == "json":
        return False
    if group(chunk) == "python" and only_comments:
        return False
    if len("".join(chunk.text.split())) < MIN_CHARS:
        return False
    if (
        len(tokenizer(build_chat_prompt(tokenizer, chunk))["input_ids"])
        > MAX_PROMPT_TOKENS
    ):
        return False
    return True


def split_files(chunks: list[Chunk], rng: random.Random) -> tuple[set[str], set[str]]:
    """
    which files go to eval and which go to training
    """
    # get the files from the chunks
    files = {c.file_path for c in chunks}
    # sort the set
    files = sorted(files)
    # shuffle to get files from the whole corpus and not
    # just from the 15% of the alphabetically first
    rng.shuffle(files)

    n_eval = round(len(files) * EVAL_FRACTION)

    eval_files = set(files[:n_eval])
    train_files = set(files[n_eval:])
    return eval_files, train_files


def sample(
    chunks: list[Chunk], quotas: dict[str, int], rng: random.Random
) -> list[Chunk]:
    """
    pick quotas[group] chunks per group from one side of the split,
    never more than MAX_PER_FILE from the same file
    """
    # sort the pool into groups
    buckets = {}
    for c in chunks:
        label = group(c)
        if label not in buckets:
            buckets[label] = []
        buckets[label].append(c)

    picked = []
    for name, quota in quotas.items():
        # copy, so shuffling doesn't reorder the caller's pool
        candidates = list(buckets.get(name, []))
        rng.shuffle(candidates)

        # walk the shuffled list instead of rng.sample, so the
        # per-file cap can be enforced while picking
        per_file = Counter()
        taken = []
        for c in candidates:
            if per_file[c.file_path] >= MAX_PER_FILE:
                continue
            taken.append(c)
            per_file[c.file_path] += 1
            if len(taken) == quota:
                break

        # fail loudly: a silently smaller split would no longer match the README
        if len(taken) < quota:
            raise ValueError(f"only {len(taken)} {name} chunks available, need {quota}")
        picked.extend(taken)
    return picked


def write(path: Path, chunks: list[Chunk]) -> None:
    """
    one record per chunk, in the format load_split reads the assistant
    turn stays empty until the chunks are labelled
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in chunks:
            record = {
                "chunk": asdict(c),  # turns the Chunk dataclass into a plain dict
                "messages": [
                    {"role": "user", "content": build_prompt(c)},
                    {"role": "assistant", "content": ""},
                ],
            }
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    chunks = chunk_corpus(Path("data/raw/vllm-0.10.1"))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    kept = [c for c in chunks if keep(c, tokenizer)]

    rng = random.Random(SEED)
    eval_files, train_files = split_files(kept, rng)
    # guardrail for files overlap if sth changes in
    # split_files in the future
    assert eval_files.isdisjoint(train_files)

    eval_pool = [c for c in kept if c.file_path in eval_files]
    train_pool = [c for c in kept if c.file_path in train_files]

    eval_chunks = sample(eval_pool, EVAL_QUOTAS, rng)
    train_chunks = sample(train_pool, TRAIN_QUOTAS, rng)

    write(Path("data/eval_unlabeled.jsonl"), eval_chunks)
    write(Path("data/train_unlabeled.jsonl"), train_chunks)

    print(f"{len(chunks)} chunks -> {len(kept)} kept")
    print(f"files: {len(eval_files)} eval, {len(train_files)} train")
    print("eval: ", dict(Counter(group(c) for c in eval_chunks)))
    print("train:", dict(Counter(group(c) for c in train_chunks)))
