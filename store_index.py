"""Build or update the Pinecone index from PDFs in the project's data folder."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.vector_store import index_documents


def main() -> None:
    load_dotenv()
    project_dir = Path(__file__).resolve().parent
    data_dir = Path(os.getenv("PDF_DATA_DIR", str(project_dir / "data")))
    if not data_dir.is_absolute():
        data_dir = project_dir / data_dir
    count = index_documents(str(data_dir))
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-bot")
    print(f"Indexed {count} text chunks into Pinecone index '{index_name}'.")


if __name__ == "__main__":
    main()
