"""Exercise the UI flow and offline fallback without network/model calls."""
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from src.chat_service import respond


CHUNKS = [{"id": "fund-1", "text": "APAC fund has currency and equity risks.",
           "metadata": {"source_path": "fund.pdf", "page": 2, "document_type": "fund_factsheet"}}]


class ChatTests(unittest.TestCase):
    def test_search_returns_original_evidence_and_abstains(self):
        with patch("src.chat_service.backend_ready", return_value=False):
            result = respond("APAC risks", CHUNKS, [], 3)
            self.assertEqual(result["sources"], CHUNKS)
            self.assertFalse(result["abstained"])
            self.assertTrue(respond("volcano", CHUNKS, [], 3)["abstained"])

    def test_backend_receives_history_and_scope(self):
        history = [{"role": "user", "content": "APAC?"}]
        with patch("src.chat_service.rag.answer", create=True, return_value={"answer": "Evidence [1]", "sources": CHUNKS}) as answer:
            self.assertEqual(respond("Risks?", CHUNKS, history, 2)["mode"], "RAG")
            answer.assert_called_once_with(question="Risks?", chunks=CHUNKS, history=history, top_k=2)

    def test_chat_sources_export_and_reset(self):
        with patch("src.chat_service.load_chunks", return_value=CHUNKS), patch("src.chat_service.backend_ready", return_value=False):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("APAC risks").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 2)
            self.assertIn("fund.pdf", app.expander[0].label)
            self.assertEqual(app.session_state["messages"][1]["sources"], CHUNKS)
            app.sidebar.button[0].click().run()
            self.assertEqual(len(app.chat_message), 0)
            self.assertFalse(app.exception)

    def test_empty_library_disables_chat(self):
        with patch("src.chat_service.load_chunks", return_value=[]):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.chat_input[0].disabled)


if __name__ == "__main__":
    unittest.main()
