
from pathlib import Path
from src.chunking.chunk_corpus.py import chunk_corpus
from src.extraction.prompts import build_prompt
from src.generate.py import build_chat_prompt

SEED = 0

def group(chunk) -> str:
    # py, docs etc

def keep(chunk, tokenizer) -> bool:
    # filters

def split_files(chunks) -> tuple[set, set]:


def sample(chunks, quotas, rng) -> list:


def write(path, chunks) -> None:



