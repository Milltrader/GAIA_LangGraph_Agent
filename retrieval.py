
import io
import pathlib
from typing import List

import pandas as pd
import requests
from utils import QUESTIONS_URL, EMBED_MODEL, get_logger
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
import json


log = get_logger(__name__)
print = log.info

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

INDEX_DIR   = pathlib.Path("index_store")
INDEX_DIR.mkdir(exist_ok=True)
FAISS_PATH  = INDEX_DIR / "gaia.faiss"
META_PATH   = INDEX_DIR / "gaia.json"


# ── Helpers ──────────────────────────────────────────────────────────
def _download_json(url: str):
    log.info(f"Downloading {url}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def _download_file(url: str, path: pathlib.Path):
    log.info(f"Downloading {url.split('/')[-1]} to {path}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    path.write_bytes(r.content)

def _sync_raw_data():
    """Pull /questions and any attachment into ./data"""
    tasks = _download_json(QUESTIONS_URL)
    json.dump(tasks, (DATA_DIR / "questions.json").open("w"))

    for t in tasks:
        if not t.get("file"):
            continue
        tid = t["task_id"]
        file_url = f"{QUESTIONS_URL.rsplit('/', 1)[0]}/files/{tid}"
        out_path = DATA_DIR / f"{tid}_{t.get('file_name', 'file')}"
        if not out_path.exists():
            _download_file(file_url, out_path)

    log.info("Raw data synced → %s files", len(list(DATA_DIR.iterdir())))


def _build_index() -> VectorStoreIndex:
    """Create and persist FAISS index on first run."""
    _sync_raw_data()

    docs = SimpleDirectoryReader(DATA_DIR).load_data(
        extra_info={"source": "gaia"}
    )  # auto-handles .csv/.png/.txt + BLIP caption for images

    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=OpenAIEmbedding(model=EMBED_MODEL),
        vector_store=FaissVectorStore(
            faiss_index_path=str(FAISS_PATH),
            faiss_config_path=str(META_PATH),
        ),
    )
    # Persist so next cold-start loads instantly
    index.storage_context.persist(INDEX_DIR)
    log.info("Index built & persisted to %s", INDEX_DIR)
    return index

def _load_index() -> VectorStoreIndex:
    """Try to load; fallback to build."""
    if FAISS_PATH.exists() and META_PATH.exists():
        log.info("Loading FAISS index from disk …")
        return VectorStoreIndex.from_vector_store(
            FaissVectorStore.load(str(FAISS_PATH), str(META_PATH))
        )
    return _build_index()


_query_engine = None

def get_query_engine(top_k: int = 3):
    """
    Returns a LlamaIndex QueryEngine (similarity_top_k = top_k)
    that the LangGraph tool can call.
    """
    global _query_engine
    if _query_engine is None:
        idx = _load_index()
        _query_engine = idx.as_query_engine(similarity_top_k=top_k)
    return _query_engine