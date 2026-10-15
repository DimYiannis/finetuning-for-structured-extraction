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
    # filters

def split_files(chunks) -> tuple[set, set]:


def sample(chunks, quotas, rng) -> list:


def write(path, chunks) -> None:


if __name__ == "__main__":
    chunks = chunk_corpus(Path("data/raw/vllm-0.10.1"))

