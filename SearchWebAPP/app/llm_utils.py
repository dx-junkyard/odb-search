"""Interact with Ollama for classification labels."""
from __future__ import annotations
import os, json, requests

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
CHAT_MODEL   = os.getenv("CHAT_MODEL", "llama3.3:latest")
API_KEY = os.getenv("API_KEY", "ollama")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "llm_service_json_prompt.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as fp:
    PROMPT_TEMPLATE = fp.read().rstrip() + "\n\n"


def label_question(question: str) -> dict:
    body = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE + question}],
        "temperature": 0.2,
    }
    r = requests.post(
        f"{OLLAMA_BASE}/chat/completions", headers=HEADERS, json=body, timeout=120
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    start, end = content.find("{"), content.rfind("}") + 1
    return json.loads(content[start:end])

