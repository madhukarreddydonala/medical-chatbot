"""Fast, provider-free smoke tests for the Flask routes."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import app


class ChatRoutesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        app.app.config.update(TESTING=True)
        self.client = app.app.test_client()

    def test_home_renders_chat_page_and_assets(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CareGuide", response.data)
        css_response = self.client.get("/static/css/chat.css")
        js_response = self.client.get("/static/js/chat.js")
        self.assertEqual(css_response.status_code, 200)
        self.assertEqual(js_response.status_code, 200)
        css_response.close()
        js_response.close()

    def test_health_is_liveness_check(self) -> None:
        self.assertEqual(self.client.get("/health").json, {"status": "ok"})

    def test_invalid_question_payloads_return_400(self) -> None:
        cases = [
            self.client.post("/api/chat", json=[]),
            self.client.post("/api/chat", json={"message": "  "}),
            self.client.post("/api/chat", json={"message": "x" * 2001}),
            self.client.post("/api/chat", data="not-json", content_type="text/plain"),
        ]
        self.assertTrue(all(response.status_code == 400 for response in cases))

    @patch("app.answer_question")
    def test_chat_returns_answer_and_sources(self, answer_question) -> None:
        answer_question.return_value = {
            "answer": "Blood pressure is the force of blood against artery walls. [1]",
            "sources": [{"id": 1, "name": "Medical_book.pdf", "page": 3}],
        }

        response = self.client.post(
            "/api/chat", json={"message": "What is blood pressure?"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["sources"][0]["page"], 3)
        answer_question.assert_called_once_with("What is blood pressure?")

    @patch("app.answer_question", side_effect=RuntimeError("Set GROQ_API_KEY."))
    def test_configuration_error_is_reported_as_service_unavailable(self, _answer) -> None:
        response = self.client.post("/api/chat", json={"message": "Question"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("GROQ_API_KEY", response.json["error"])


if __name__ == "__main__":
    unittest.main()
