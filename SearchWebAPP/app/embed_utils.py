"""BERT‑based text embedding for similarity search."""
import torch, numpy as np
from transformers import BertTokenizer, BertModel
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model

@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    tokenizer, model = _load_bert()
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return emb.numpy()
