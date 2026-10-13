"""Provider-free tests for retrieval formatting and source citations."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.rag import answer_question


class RagAnswerTestCase(unittest.TestCase):
    @patch("src.rag._get_qa_components")
    def test_no_retrieved_documents_returns_no_answer(self, get_components) -> None:
        retriever = type("Retriever", (), {"invoke": lambda self, _query: []})()
        answer_chain = type("Chain", (), {"invoke": lambda self, _inputs: self.fail("should not invoke")})()
        get_components.return_value = (retriever, answer_chain)

        result = answer_question("A question outside the library")

        self.assertEqual(result["sources"], [])
        self.assertIn("couldn't find relevant information", result["answer"])

    @patch("src.rag._get_qa_components")
    def test_answer_has_page_source_and_context_marker(self, get_components) -> None:
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
        get_components.return_value = (retriever, answer_chain)

        result = answer_question("What are the two blood pressure numbers?")

        self.assertEqual(result["answer"], "A reading has two numbers. [1]")
        self.assertEqual(result["sources"], [{"id": 1, "name": "Medical_book.pdf", "page": 3}])
        self.assertIn("[1]", received["context"])
        self.assertIn("Blood pressure is measured", received["context"])


if __name__ == "__main__":
    unittest.main()
