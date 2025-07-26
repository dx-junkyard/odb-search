"""OpenAI埋め込みAPIによるテキスト埋め込み生成ユーティリティ"""
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

def embed_text(text: str) -> np.ndarray:
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)
