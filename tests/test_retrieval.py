import json
import tempfile
import unittest
from pathlib import Path

import chromadb

from src.retrieval import VectorRetriever, load_chunks, retrieval_metrics


class FakeEncoder:
    model_name = "test-keyword-encoder"

    def encode(self, texts, *, batch_size=32):
        vectors = []
        for text in texts:
            value = text.lower()
            vectors.append([
                float("risk" in value or "suitable" in value),
                float("bond" in value or "income" in value),
                0.01,
            ])
        return vectors


class RetrievalTests(unittest.TestCase):
    def _write_chunks(self, path):
        rows = [
            {
                "id": "risk-policy",
                "text": "The suitability policy defines risk matching requirements.",
                "metadata": {"source_path": "policy/risk.pdf", "page": 2},
            },
            {
                "id": "bond-fact-sheet",
                "text": "The bond income fund invests in global fixed income.",
                "metadata": {"source_path": "funds/bond.pdf", "page": 1},
            },
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def test_load_chunks_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "chunks.jsonl"
            row = {"id": "same", "text": "text", "metadata": {}}
            path.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate chunk id"):
                load_chunks(path)

    def test_index_is_repeatable_and_search_preserves_citations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            chunks = root / "chunks.jsonl"
            self._write_chunks(chunks)
            retriever = VectorRetriever(
                root / "db",
                "test_collection",
                FakeEncoder(),
                client=chromadb.EphemeralClient(),
            )
            self.assertEqual(retriever.index(chunks).chunks_indexed, 2)
            self.assertEqual(retriever.index(chunks).chunks_indexed, 2)
            self.assertEqual(retriever.collection.count(), 2)
            result = retriever.search("Is this suitable for the client's risk profile?", top_k=1)[0]
            self.assertEqual(result["id"], "risk-policy")
            self.assertEqual(result["metadata"]["source_path"], "policy/risk.pdf")
            self.assertEqual(result["metadata"]["page"], 2)

    def test_source_level_metrics(self):
        results = [
            {"metadata": {"source_path": "a.pdf"}},
            {"metadata": {"source_path": "irrelevant.pdf"}},
        ]
        metrics = retrieval_metrics(results, ["a.pdf", "b.pdf"])
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["hit"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 1.0)


if __name__ == "__main__":
    unittest.main()
