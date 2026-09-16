"""Lexical diagnostics only: these checks cannot establish factual entailment."""
import re
from collections.abc import Mapping


def normalize_text(text):
    """Convert text into a simple set of words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return set(text.split())


def check_hallucination(answer, retrieved_documents, threshold=0.25):
    """Accept strings or pipeline evidence dictionaries; flag lexical problems.

    Legacy keys are retained, but ``supported=False`` means NOT VERIFIED,
    not necessarily false. ``confidence`` is zero because no factual verifier
    runs here. Never use this helper alone to approve or reject an LLM answer.
    Threshold applies only to lexical overlap, not factual confidence.
    """
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if answer is None:
        answer = ""
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    documents = retrieved_documents if retrieved_documents is not None else []
    if isinstance(documents, (str, Mapping)):
        documents = [documents]
    texts = []
    labels = set()
    for index, document in enumerate(documents, 1):
        value = document.get("text", "") if isinstance(document, Mapping) else document
        if not isinstance(value, str):
            raise TypeError("Evidence must be text or a dictionary with string text")
        if value.strip():
            texts.append(value)
            labels.add(str(document.get("label", f"S{index}")) if isinstance(document, Mapping) else f"S{index}")
    clean_answer = re.sub(r"\[S\d+\]", "", answer)
    words = normalize_text(clean_answer)
    context = " ".join(texts)
    overlap = len(words & normalize_text(context)) / len(words) if words else 0.0
    issues = []
    if not words or not texts:
        issues.append("insufficient_evidence_or_empty_answer")
    if overlap < threshold:
        issues.append("low_lexical_overlap")
    invalid = sorted(set(re.findall(r"\[(S\d+)\]", answer)) - labels)
    if invalid:
        issues.append("unknown_citation_labels")
    def numbers(text):
        return set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d)", "", text)))
    missing_numbers = sorted(numbers(clean_answer) - numbers(context))
    if missing_numbers:
        issues.append("numbers_absent_from_evidence")
    return {"supported": False, "confidence": 0.0, "status": "unverified",
            "message": "Lexical diagnostics only; factual support requires claim-level verification.",
            "lexical_overlap": round(overlap, 4), "issues": issues,
            "unknown_citations": invalid, "numbers_absent_from_evidence": missing_numbers}
