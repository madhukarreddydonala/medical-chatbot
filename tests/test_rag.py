"""Provider-free tests for retrieval formatting and source citations."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.rag import answer_question


class RagAnswerTestCase(unittest.TestCase):
    @patch("src.rag._get_retriever")
    def test_no_retrieved_documents_returns_no_answer(self, get_retriever) -> None:
        retriever = type("Retriever", (), {"invoke": lambda self, _query: []})()
        get_retriever.return_value = retriever

        result = answer_question("A question outside the library", "test-key")

        self.assertEqual(result["sources"], [])
        self.assertIn("couldn't find relevant information", result["answer"])

    @patch("src.rag._get_answer_chain")
    @patch("src.rag._get_retriever")
    def test_answer_has_page_source_and_context_marker(self, get_retriever, get_answer_chain) -> None:
        documents = [
            Document(
                page_content="Blood pressure is measured in two numbers.",
                metadata={"source": "C:/library/Medical_book.pdf", "page": 2},
            ),
            Document(
                page_content="The upper number is systolic.",
                metadata={"source": "C:/library/Medical_book.pdf", "page": 2},
            ),
        ]
        retriever = type("Retriever", (), {"invoke": lambda self, _query: documents})()
        received = {}

        def fake_invoke(inputs):
            received.update(inputs)
            return "A reading has two numbers. [1]"

        answer_chain = type("Chain", (), {"invoke": lambda self, inputs: fake_invoke(inputs)})()
        get_retriever.return_value = retriever
        get_answer_chain.return_value = answer_chain

        result = answer_question("What are the two blood pressure numbers?", "test-key")

        self.assertEqual(result["answer"], "A reading has two numbers. [1]")
        self.assertEqual(result["sources"], [{"id": 1, "name": "Medical_book.pdf", "page": 3}])
        self.assertIn("[1]", received["context"])
        self.assertIn("Blood pressure is measured", received["context"])


if __name__ == "__main__":
    unittest.main()
