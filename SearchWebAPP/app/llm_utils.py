"""Interact with Ollama for label classification and service selection."""
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

SELECT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "llm_service_select_prompt.txt")
try:
    with open(SELECT_PROMPT_PATH, "r", encoding="utf-8") as fp:
        SELECT_PROMPT_TEMPLATE = fp.read().rstrip() + "\n\n"
except FileNotFoundError:
    SELECT_PROMPT_TEMPLATE = ""


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


class ServiceSelector:
    """Select best services from candidates using LLM."""

    def __init__(self, max_select: int = 3):
        self.max_select = max_select

    def recommend(self, question: str, candidates: list[dict]) -> list[dict]:
        payload = {
            "question": question,
            "candidates": candidates,
        }
        body = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": SELECT_PROMPT_TEMPLATE + json.dumps(payload, ensure_ascii=False)}],
            "temperature": 0.2,
        }
        r = requests.post(
            f"{OLLAMA_BASE}/chat/completions", headers=HEADERS, json=body, timeout=120
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        start, end = content.find("{"), content.rfind("}") + 1
        try:
            data = json.loads(content[start:end])
            recs = data.get("recommendations", [])
            return recs[: self.max_select]
        except Exception:
            return []

