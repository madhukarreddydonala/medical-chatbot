"""Retrieval-augmented question answering using Pinecone and Groq."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser

from src.prompt import ANSWER_PROMPT
from src.vector_store import get_vector_store


@lru_cache(maxsize=1)
def _get_qa_components():
    """Build and cache Pinecone retrieval and the Groq-backed answer chain."""
    required = [
        key
        for key in ("PINECONE_API_KEY", "GROQ_API_KEY")
        if not os.getenv(key)
    ]
    if required:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(required)
        )

    from langchain_groq import ChatGroq

    vector_store = get_vector_store()
    try:
        result_count = int(os.getenv("RETRIEVER_K", "4"))
    except ValueError as exc:
        raise RuntimeError("RETRIEVER_K must be an integer from 1 to 12.") from exc
    if not 1 <= result_count <= 12:
        raise RuntimeError("RETRIEVER_K must be an integer from 1 to 12.")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": result_count},
    )
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.1,
        max_tokens=700,
        reasoning_format="hidden",
        timeout=45,
        max_retries=2,
    )
    return retriever, ANSWER_PROMPT | llm | StrOutputParser()


def _source_details(documents) -> list[dict[str, str | int]]:
    sources: list[dict[str, str | int]] = []
    seen: set[tuple[str, int]] = set()
    for document in documents:
        metadata = document.metadata or {}
        source_path = str(metadata.get("source", "Medical library"))
        source_name = Path(source_path.replace("\\", "/")).name or "Medical library"
        try:
            page = int(metadata.get("page", 0)) + 1
        except (TypeError, ValueError):
            page = 1
        key = (source_name, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"id": len(sources) + 1, "name": source_name, "page": page})
    return sources


def answer_question(question: str) -> dict[str, object]:
    """Retrieve relevant PDF passages and answer with source references."""
    retriever, answer_chain = _get_qa_components()
    documents = retriever.invoke(question)
    if not documents:
        return {
            "answer": "I couldn't find relevant information in the medical library for that question.",
            "sources": [],
        }

    sources = _source_details(documents)
    source_ids: dict[tuple[str, int], int] = {
        (str(source["name"]), int(source["page"])): int(source["id"])
        for source in sources
    }
    excerpts: list[str] = []
    for document in documents:
        metadata = document.metadata or {}
        name = Path(str(metadata.get("source", "Medical library")).replace("\\", "/")).name
        try:
            page = int(metadata.get("page", 0)) + 1
        except (TypeError, ValueError):
            page = 1
        citation_id = source_ids.get((name or "Medical library", page), 1)
        excerpts.append(f"[{citation_id}]\n{document.page_content}")

    answer = answer_chain.invoke(
        {"question": question, "context": "\n\n".join(excerpts)}
    )
    return {"answer": answer, "sources": sources}
