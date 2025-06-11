"""
GAIA LangGraph Agent - Document Retrieval Module

This module handles document indexing and retrieval using LlamaIndex and FAISS.
It supports various file types including text, images, audio, and documents.
"""

import io
import pathlib
import json
import shutil
from typing import List

import pandas as pd
import requests
import faiss
from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
    Document
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from utils import QUESTIONS_URL, EMBED_MODEL, get_logger

# Initialize logging
log = get_logger(__name__)
print = log.info

# File type definitions
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg"}
DOC_EXT = {".xlsx", ".py"}
TEXT_EXT = {".txt", ".csv", ".json", ".pdf"}

# Directory setup
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

INDEX_DIR = pathlib.Path("index_store")
INDEX_DIR.mkdir(exist_ok=True)
FAISS_PATH = INDEX_DIR / "gaia.faiss"
META_PATH = INDEX_DIR / "gaia.json"

def _download_json(url: str) -> dict:
    """Download and parse JSON from URL."""
    log.info(f"Downloading {url}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def _download_file(url: str, path: pathlib.Path) -> None:
    """Download file from URL and save to specified path."""
    log.info(f"Downloading {url.split('/')[-1]} to {path}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    path.write_bytes(r.content)

def _sync_raw_data() -> None:
    """
    Synchronize data from the questions API.
    Downloads questions and associated files to the data directory.
    """
    tasks = _download_json(QUESTIONS_URL)
    json.dump(tasks, (DATA_DIR / "questions.json").open("w"))

    for task in tasks:
        fname = task.get("file_name")             
        if not fname:                          
            continue
        task_id = task["task_id"]
        url = f"{QUESTIONS_URL.rsplit('/',1)[0]}/files/{task_id}"
        path = DATA_DIR / f"{task_id}_{fname}"
        if not path.exists():
            _download_file(url, path)

    log.info("Raw data synced → %s files", len(list(DATA_DIR.iterdir())))

def _build_index() -> VectorStoreIndex:
    """
    Build an in-memory FAISS index from available documents.
    Handles both text files and binary files (images, audio, documents).
    """
    _sync_raw_data()

    # 1) Process text files
    text_files = [
        f for f in Path(DATA_DIR).iterdir() if f.suffix.lower() in TEXT_EXT
    ]

    reader_docs: list[Document] = []
    if text_files:
        reader_docs = SimpleDirectoryReader(
            input_files=[str(p) for p in text_files]
        ).load_data()

    # 2) Add binary files as file paths
    extra_docs: list[Document] = []
    for file in Path(DATA_DIR).iterdir():
        suf = file.suffix.lower()
        if suf in IMAGE_EXT | AUDIO_EXT | DOC_EXT:
            extra_docs.append(
                Document(
                    page_content=file.as_posix(),
                    metadata={"filetype": suf}
                )
            )

    docs = reader_docs + extra_docs

    # 3) Create FAISS index
    embed_dim = 1536  # OpenAI embedding dimension
    index_flat = faiss.IndexFlatL2(embed_dim)
    vector_store = FaissVectorStore(faiss_index=index_flat)

    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=OpenAIEmbedding(model=EMBED_MODEL),
        vector_store=vector_store,
    )
    log.info("FAISS index built (%d docs).", len(docs))
    return index

def _load_index() -> VectorStoreIndex:
    """Load or build the document index."""
    return _build_index()

# Global query engine instance
_query_engine = None

def get_query_engine(top_k: int = 3) -> VectorStoreIndex:
    """
    Get or create a LlamaIndex QueryEngine for document retrieval.
    
    Args:
        top_k: Number of most similar documents to retrieve
        
    Returns:
        A configured QueryEngine instance
    """
    global _query_engine
    if _query_engine is None:
        idx = _load_index()
        _query_engine = idx.as_query_engine(similarity_top_k=top_k)
    return _query_engine

