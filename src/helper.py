"""PDF loading, chunking, and embedding helpers for the RAG pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def load_pdf(path: str | Path) -> list[Document]:
    """Load every PDF under ``path`` and preserve filename and page metadata."""
    from langchain_community.document_loaders import PyPDFLoader

    data_path = Path(path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {data_path}")
    pdf_paths = sorted(data_path.rglob("*.pdf")) if data_path.is_dir() else [data_path]
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_path}")

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        documents.extend(PyPDFLoader(str(pdf_path)).load())
    return [doc for doc in documents if doc.page_content.strip()]


def text_split(
    documents: list[Document], *, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Document]:
    """Split pages into searchable chunks while retaining source/page metadata."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


@lru_cache(maxsize=1)
def download_hugging_face_embeddings():
    """Create the configured local sentence-transformer embeddings once."""
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=model_name)
