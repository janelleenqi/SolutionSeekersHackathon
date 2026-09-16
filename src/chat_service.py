"""UI adapter and offline evidence search; no model credentials required."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from src import rag

STOPWORDS = set("a an the is are was were to of in on for with and or my me what which who how can do does this that about please tell show".split())


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        return []
    chunks = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not isinstance(item.get("metadata"), dict) or not item.get("id"):
            raise ValueError(f"Invalid chunk on line {number}.")
        chunks.append(item)
    return chunks


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def backend_ready() -> bool:
    return callable(getattr(rag, "answer", None))


def respond(question: str, chunks: list[dict], history: list[dict], top_k: int) -> dict:
    """RAG contract: answer(question=, chunks=, history=, top_k=) -> dict.

    Return answer (str), sources (list of chunk dicts), and abstained (bool).
    The backend owns retrieval, grounding, and abstention decisions.
    """
    if backend_ready():
        result = rag.answer(question=question, chunks=chunks, history=history, top_k=top_k)
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
            raise ValueError("The RAG backend must return a dictionary containing an answer string.")
        sources = result.get("sources", [])
        if not isinstance(sources, list) or any(
            not isinstance(s, dict) or not isinstance(s.get("text"), str)
            or not isinstance(s.get("metadata"), dict) for s in sources
        ):
            raise ValueError("The RAG backend returned invalid source passages.")
        return {**result, "sources": sources, "mode": "RAG"}

    query = tokens(question)
    documents = [tokens(chunk["text"]) for chunk in chunks]
    frequency = Counter(term for document in documents for term in document)
    ranked = []
    for chunk, document in zip(chunks, documents):
        overlap = query & document
        if overlap:
            score = sum(math.log(1 + len(chunks) / frequency[t]) for t in overlap)
            ranked.append((score, chunk))
    sources = [chunk for _, chunk in sorted(ranked, key=lambda item: item[0], reverse=True)[:top_k]]
    if not sources:
        answer = "I couldn't find matching evidence in the selected documents. Try a client ID, fund name, or a more specific question, or add the relevant document."
    else:
        answer = "I found the following potentially relevant passages. Open the sources below to review the evidence. Document-search mode does not generate an answer or determine whether these passages fully support your question."
    return {"answer": answer, "sources": sources, "abstained": not sources, "mode": "Document search"}
