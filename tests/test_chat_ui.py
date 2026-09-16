"""Exercise the UI flow and offline fallback without network/model calls."""
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


CHUNKS = [{"id": "fund-1", "text": "APAC fund has currency and equity risks.",
           "metadata": {"source_path": "fund.pdf", "page": 2, "document_type": "fund_factsheet"}}]


class ChatTests(unittest.TestCase):
    def test_chat_sources_export_and_reset(self):
        sources = [{**CHUNKS[0], "label": "S1", "cited": True, "score": 0.8}]
        result = {"answer": "Currency risks [S1].", "sources": sources,
                  "mode": "Hybrid RAG", "abstained": False, "evidence_count": 1}
        with patch("src.chat_service.load_chunks", return_value=CHUNKS), patch("src.chat_service.backend_ready", return_value=True), patch("src.chat_service.respond", return_value=result) as respond:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("APAC risks").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 2)
            self.assertIn("fund.pdf", app.expander[0].label)
            self.assertEqual(app.session_state["messages"][1]["sources"], sources)
            respond.assert_called_once_with("APAC risks", top_k=5, document_types=["fund_factsheet"], preview=False)
            self.assertEqual(len(app.get("download_button")), 1)
            app.sidebar.button[0].click().run()
            self.assertEqual(len(app.chat_message), 0)
            self.assertFalse(app.exception)

    def test_empty_library_disables_chat(self):
        with patch("src.chat_service.load_chunks", return_value=[]), patch("src.chat_service.backend_ready", return_value=False):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.chat_input[0].disabled)

    def test_missing_configuration_uses_preview(self):
        result = {"answer": "Review evidence.", "sources": [], "mode": "Evidence preview", "evidence_count": 0}
        with patch("src.chat_service.load_chunks", return_value=CHUNKS), patch("src.chat_service.backend_ready", return_value=False), patch("src.chat_service.respond", return_value=result) as respond:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            self.assertTrue(app.sidebar.toggle[0].value)
            app.chat_input[0].set_value("APAC risks").run()
            self.assertFalse(app.exception)
            self.assertTrue(respond.call_args.kwargs["preview"])

    def test_backend_error_is_displayed(self):
        with patch("src.chat_service.load_chunks", return_value=CHUNKS), patch("src.chat_service.backend_ready", return_value=True), patch("src.chat_service.respond", side_effect=RuntimeError("Provider unavailable")):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            app.chat_input[0].set_value("APAC risks").run()
            self.assertFalse(app.exception)
            self.assertIn("Provider unavailable", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
