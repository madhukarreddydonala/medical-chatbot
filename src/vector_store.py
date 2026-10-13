"""Pinecone index creation, PDF ingestion, and vector-store connection."""

from __future__ import annotations

import hashlib
import os
import time

from src.helper import EMBEDDING_DIMENSION, download_hugging_face_embeddings, load_pdf, text_split


def _pinecone_client():
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in your .env file before indexing.")
    return Pinecone(api_key=api_key)


def _index_settings() -> tuple[str, str, str]:
    name = os.getenv("PINECONE_INDEX_NAME", "medical-bot").strip()
    cloud = os.getenv("PINECONE_CLOUD", "aws").strip()
    region = os.getenv("PINECONE_REGION", "us-east-1").strip()
    if not name or not cloud or not region:
        raise RuntimeError("Pinecone index name, cloud, and region must be set.")
    return name, cloud, region


def _ensure_index(client, name: str, cloud: str, region: str) -> None:
    from pinecone import ServerlessSpec

    if not client.has_index(name):
        client.create_index(
            name=name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        deadline = time.monotonic() + 300
        while not client.describe_index(name).status.get("ready", False):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Pinecone index {name!r} did not become ready in time.")
            time.sleep(2)

    description = client.describe_index(name)
    if description.dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Index {name!r} has dimension {description.dimension}, but the configured "
            f"embedding model requires {EMBEDDING_DIMENSION}. Use a new index name."
        )


def _stable_id(document, ordinal: int) -> str:
    metadata = document.metadata or {}
    source = str(metadata.get("source", "unknown"))
    page = str(metadata.get("page", 0))
    start = str(metadata.get("start_index", ordinal))
    return hashlib.sha256(f"{source}:{page}:{start}".encode("utf-8")).hexdigest()


def index_documents(data_dir: str = "data/") -> int:
    """Load local PDFs, create the index if needed, and upsert their chunks."""
    client = _pinecone_client()
    index_name, cloud, region = _index_settings()
    documents = load_pdf(data_dir)
    chunks = text_split(documents)
    if not chunks:
        raise RuntimeError(f"No readable text was found in PDF directory {data_dir!r}.")

    embeddings = download_hugging_face_embeddings()
    _ensure_index(client, index_name, cloud, region)

    from langchain_pinecone import PineconeVectorStore

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=os.getenv("PINECONE_NAMESPACE") or None,
        ids=[_stable_id(document, ordinal) for ordinal, document in enumerate(chunks)],
    )
    return len(chunks)


def get_vector_store():
    """Connect to an already-populated Pinecone index for search."""
    from langchain_pinecone import PineconeVectorStore

    client = _pinecone_client()
    index_name, _cloud, _region = _index_settings()
    if not client.has_index(index_name):
        raise RuntimeError(
            f"Pinecone index {index_name!r} does not exist. Run `uv run python store_index.py` first."
        )
    description = client.describe_index(index_name)
    if description.dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Index {index_name!r} uses dimension {description.dimension}; "
            f"expected {EMBEDDING_DIMENSION}."
        )
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=download_hugging_face_embeddings(),
        namespace=os.getenv("PINECONE_NAMESPACE") or None,
    )
