"""
    script that picks which chunks go into the dataset
"""
import json
import random
from dataclasses import asdict
from transformers import AutoTokenizer

from src.chunking.spans import Chunk
from pathlib import Path
from src.chunking.chunk_corpus import chunk_corpus
from src.extraction.prompts import build_prompt
from src.generate import build_chat_prompt

SEED = 0
MIN_CHARS = 100
MAX_PROMPT_TOKENS = 1700

def group(chunk) -> str:
    """
        sort chunks by file extension
    """
    suffix = Path(chunk.file_path).suffix.lower()
    if suffix == ".py":
        return "python"
    elif suffix in {".md", ".rst", ".txt"}:
        return "text"
    elif suffix == ".json":
        return "json"
    else:
        return "other"


def keep(chunk, tokenizer) -> bool:
    """
        determine whether a chunk
        is worth putting in the dataset
    """
    # check if its only comments
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    if all(line.startswith("#") for line in lines):
        only_comments = True

    if group(chunk) == "json":
        return False
    if group(chunk) == "python" and only_commmets:
        return False
    if len("".join(chunk.text.split())) < MIN_CHARS:
        return False
    if len(tokenizer(build_chat_prompt(tokenizer, chunk))["input_ids"]) > MAX_PROMPT_TOKENS:
        return False
    return True


def split_files(chunks) -> tuple[set, set]:


def sample(chunks, quotas, rng) -> list:


def write(path, chunks) -> None:


if __name__ == "__main__":
    chunks = chunk_corpus(Path("data/raw/vllm-0.10.1"))
    
    buckets = {}
    for c in chunks:
        label = group(c)
        if label not in buckets:
            buckets[label] = []          # first chunk of this group: make its list
        buckets[label].append(c)
    AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").
    kept = [c for c in chunks if keep(c, tokenizer)]

