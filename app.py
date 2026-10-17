"""Flask web application for the CareGuide medical information assistant."""

from __future__ import annotations

import hmac
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from src.rag import answer_question

load_dotenv()

app = Flask(__name__, static_folder="public/static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
app.logger.setLevel(logging.INFO)


@app.after_request
def add_security_headers(response):
    """Apply browser protections to the page and JSON API responses."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'self'; frame-ancestors 'none'",
    )
    return response


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
    """Validate a question and use its visitor-supplied Groq key for one answer."""
    proxy_token = os.getenv("NETLIFY_PROXY_TOKEN", "")
    require_proxy = os.getenv("REQUIRE_NETLIFY_PROXY", "0").strip() == "1"
    if require_proxy and len(proxy_token) < 32:
        app.logger.error("Netlify proxy authentication is required but not configured securely.")
        return jsonify({"error": "The chat service is not configured yet."}), 503
    if proxy_token and not hmac.compare_digest(
        request.headers.get("X-Netlify-Proxy-Token", ""), proxy_token
    ):
        return jsonify({"error": "This endpoint only accepts requests through the site."}), 403

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Send a JSON object containing a message."}), 400

    question = body.get("message")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Enter a question to get started."}), 400
    question = question.strip()
    if len(question) > 2000:
        return jsonify({"error": "Please keep your question under 2,000 characters."}), 400

    groq_api_key = request.headers.get("X-Groq-API-Key", "").strip()
    if not groq_api_key or len(groq_api_key) > 512:
        return jsonify({"error": "Enter a valid Groq API key to continue."}), 400

    try:
        result = answer_question(question, groq_api_key=groq_api_key)
        return jsonify(result)
    except RuntimeError as exc:
        app.logger.warning("Chat service configuration issue: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Chat request failed")
        return jsonify({"error": "The assistant is temporarily unavailable. Please try again shortly."}), 503


if __name__ == "__main__":
    # Flask's built-in server is for local development only. Use a production
    # WSGI server behind a TLS-terminating proxy for a deployed app.
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1" and os.getenv("HOST", "127.0.0.1") in {"127.0.0.1", "localhost"},
    )
