"""
    script that picks which chunks go into the dataset
"""
import json
import random
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
    if len(tokenizer(build_chat_prompt(tokenizer, chunk))["input_ids"]) > MAX_PROMPT_TOKENS:
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

def sample(chunks: list[Chunk], quotas: dict[str, int], rng: random.Random) -> list[Chunk]:


def write(path: Path, chunks: list[Chunk]) -> None:


if __name__ == "__main__":
    chunks = chunk_corpus(Path("data/raw/vllm-0.10.1"))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    kept = [c for c in chunks if keep(c, tokenizer)]

    rng = random.Random(SEED)
    eval_files, train_files = split_files(kept, rng)

    buckets = {}
    for c in kept:
        label = group(c)
        if label not in buckets:
            buckets[label] = []          # first chunk of this group: make its list
        buckets[label].append(c)

