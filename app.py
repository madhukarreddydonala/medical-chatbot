"""Flask application for the medical document question-answering chatbot."""

import os
from functools import lru_cache
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings

load_dotenv()

app = Flask(__name__)


@lru_cache(maxsize=1)
def get_retrieval_chain():
    """Build the retriever and model once, when the first question arrives."""
    required = [name for name in ("PINECONE_API_KEY", "OPENAI_API_KEY") if not os.getenv(name)]
    if required:
        raise RuntimeError(f"Missing environment variable(s): {', '.join(required)}")

    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-bot")
    embeddings = download_hugging_face_embeddings()
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful medical information assistant. Use only the retrieved context below. If it does not answer the question, say you do not know. Keep the response concise, use at most three sentences, and remind users to consult a qualified clinician for personal medical decisions.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    return retriever, llm, prompt


@app.get("/")
def home():
    return render_template("chat.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "Send a JSON object containing a message."}), 400
    question = body.get("message", "")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Enter a question to get started."}), 400
    if len(question) > 2000:
        return jsonify({"error": "Please keep your question under 2,000 characters."}), 400

    try:
        retriever, llm, prompt = get_retrieval_chain()
        documents = retriever.invoke(question.strip())
        context = "\n\n".join(doc.page_content for doc in documents)
        answer = llm.invoke(prompt.format_messages(context=context, question=question.strip()))
        return jsonify({"answer": answer.content})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Chat request failed")
        return jsonify({"error": "The assistant could not answer right now. Check the server configuration and try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
