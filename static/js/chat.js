"use strict";

const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const conversation = document.getElementById("conversation");
const characterCount = document.getElementById("character-count");
const MAX_CHARS = 2000;

function scrollToLatest() {
  conversation.scrollTop = conversation.scrollHeight;
}

function makeAvatar(isUser) {
  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = isUser ? "You" : "+";
  return avatar;
}

function addMessage(text, role, options = {}) {
  document.getElementById("welcome")?.remove();

  const row = document.createElement("div");
  row.className = `message ${role}${options.error ? " error" : ""}`;

  const body = document.createElement("div");
  body.className = "message-body";

  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "CareGuide";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  body.append(label, bubble);
  row.append(makeAvatar(role === "user"), body);

  if (Array.isArray(options.sources) && options.sources.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    sourceList.setAttribute("aria-label", "Answer sources");

    for (const source of options.sources) {
      if (!source || typeof source.name !== "string") continue;
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = `[${source.id}] ${source.name}${Number.isInteger(source.page) ? ` · p. ${source.page}` : ""}`;
      sourceList.append(chip);
    }
    if (sourceList.childElementCount) body.append(sourceList);
  }

  conversation.append(row);
  scrollToLatest();
  return row;
}

function addPendingMessage() {
  const row = addMessage("Searching the library", "assistant");
  const bubble = row.querySelector(".bubble");
  bubble.classList.add("typing");
  bubble.textContent = "Searching the library";
  const dots = document.createElement("span");
  dots.className = "typing-dots";
  dots.setAttribute("aria-hidden", "true");
  bubble.append(dots);
  return row;
}

async function ask(questionText) {
  const message = questionText.trim();
  if (!message || sendButton.disabled) return;

  addMessage(message, "user");
  input.value = "";
  input.style.height = "";
  updateCharacterCount();
  sendButton.disabled = true;
  const pending = addPendingMessage();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ message }),
    });
    const result = await response.json().catch(() => ({}));
    pending.remove();

    if (!response.ok) {
      throw new Error(result.error || "The assistant could not answer. Please try again.");
    }
    addMessage(
      typeof result.answer === "string" && result.answer.trim()
        ? result.answer
        : "I couldn't find an answer in the medical library.",
      "assistant",
      { sources: result.sources },
    );
  } catch (error) {
    pending.remove();
    addMessage(
      error instanceof Error ? error.message : "Unable to reach the assistant. Check your connection and try again.",
      "assistant",
      { error: true },
    );
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

function updateCharacterCount() {
  characterCount.textContent = `${input.value.length.toLocaleString()} / ${MAX_CHARS.toLocaleString()}`;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(input.value);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 132)}px`;
  updateCharacterCount();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => ask(button.dataset.question || ""));
});

updateCharacterCount();
