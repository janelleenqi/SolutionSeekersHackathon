import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion import IngestionConfig, chunk_text, clean_text, run_ingestion


class IngestionTests(unittest.TestCase):
    def test_clean_text_repairs_pdf_hyphenation(self):
        self.assertEqual(clean_text("invest-\nment   policy\r\n\r\n\r\nNext"), "investment policy\n\nNext")

    def test_chunking_is_bounded_and_overlapping(self):
        text = "\n\n".join(f"Paragraph {i} has useful evidence and details." for i in range(20))
        chunks = chunk_text(text, chunk_size=180, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 230 for chunk in chunks))
        self.assertTrue(any(part in chunks[1] for part in chunks[0].split("\n\n")[-2:]))

    def test_correspondence_metadata_and_ids_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, out = root / "raw", root / "processed"
            raw.mkdir()
            payload = {"email_threads": [{
                "thread_id": "EML-001", "subject": "Suitability review",
                "related_client_id": "CL002", "messages": [{
                    "from": "rm@example.com", "to": "client@example.com",
                    "date": "2026-01-20", "body": "Documented client comprehension concern."
                }]
            }]}
            (raw / "client_correspondence.json").write_text(json.dumps(payload), encoding="utf-8")
            config = IngestionConfig(raw, out, chunk_size=300, chunk_overlap=40)
            self.assertEqual(run_ingestion(config)["total_chunks"], 1)
            first = json.loads((out / "chunks.jsonl").read_text().splitlines()[0])
            self.assertEqual(run_ingestion(config)["total_chunks"], 1)
            second = json.loads((out / "chunks.jsonl").read_text().splitlines()[0])
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(first["metadata"]["client_id"], "CL002")
            self.assertEqual(first["metadata"]["record_id"], "EML-001-M1")


if __name__ == "__main__":
    unittest.main()
