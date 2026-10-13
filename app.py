"""Flask web application for the CareGuide medical information assistant."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from src.rag import answer_question

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
app.logger.setLevel(logging.INFO)


@app.get("/")
def home():
    """Render the chat interface."""
    return render_template("chat.html")


@app.get("/health")
def health():
    """Return an application liveness response without contacting providers."""
    return jsonify({"status": "ok"})


@app.post("/api/chat")
def chat():
    """Validate one question and return a grounded answer with source metadata."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Send a JSON object containing a message."}), 400

    question = body.get("message")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Enter a question to get started."}), 400
    question = question.strip()
    if len(question) > 2000:
        return jsonify({"error": "Please keep your question under 2,000 characters."}), 400

    try:
        result = answer_question(question)
        return jsonify(result)
    except RuntimeError as exc:
        app.logger.warning("Chat service configuration issue: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Chat request failed")
        return jsonify({"error": "The assistant is temporarily unavailable. Please try again shortly."}), 503


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
