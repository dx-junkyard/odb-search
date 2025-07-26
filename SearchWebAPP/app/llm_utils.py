"""OpenAI APIを用いたラベル分類とサービス選択"""
from __future__ import annotations
import os, json, logging
from openai import OpenAI

logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
client = OpenAI(api_key=API_KEY)

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
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE + question}
    ]
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=120,
        )
        content = response.choices[0].message.content
        start, end = content.find("{"), content.rfind("}") + 1
        return json.loads(content[start:end])
    except Exception:
        logger.exception("Failed to request label classification from OpenAI API")
        raise

class ServiceSelector:
    """LLMを用いたサービス推薦"""
    def __init__(self, max_select: int = 3):
        self.max_select = max_select

    def recommend(self, question: str, candidates: list[dict]) -> list[dict]:
        payload = {
            "question": question,
            "candidates": candidates,
        }
        messages = [
            {"role": "user", "content": SELECT_PROMPT_TEMPLATE + json.dumps(payload, ensure_ascii=False)}
        ]
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                timeout=120,
            )
            content = response.choices[0].message.content
            start, end = content.find("{"), content.rfind("}") + 1
            data = json.loads(content[start:end])
            recs = data.get("recommendations", [])
            return recs[: self.max_select]
        except Exception:
            logger.exception("Failed to request service selection from OpenAI API")
            return []

