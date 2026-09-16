"""Real SQLite execution with deterministic model responses; no API calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src import chat_service, rag
from src.database import load_database
from src.structured_data import StructuredDataStore


class SqlIntegrationTests(unittest.TestCase):
    def test_real_database_matches_original_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wealth.db'
            load_database(Path('data/raw'), path)
            store = StructuredDataStore()
            planner = Mock()
            planner.complete.return_value = json.dumps({'sql': 'SELECT client_id, name, risk_score FROM clients WHERE client_id=?', 'parameters': ['CL002']})
            router = chat_service.FallbackSqlRouter(rag.SqlStructuredRouter(planner, path), rag.StructuredQueryRouter(store))
            evidence = router.search('Robert Chua risk profile')
            self.assertEqual(evidence, router.search('Robert Chua risk profile'))
            planner.complete.assert_called_once()
            self.assertFalse(router.used_fallback)
            rows = json.loads(evidence[0]['text'].split('Query result: ')[1])
            client = next(c for c in store.clients if c['client_id'] == 'CL002')
            self.assertEqual(rows[0]['risk_score'], client['risk_score_1_to_10'])
            self.assertEqual(evidence[0]['metadata']['source_references'][0]['parameters'], ['CL002'])

    def test_failed_and_empty_sql_fall_back_once(self):
        for outcome in [RuntimeError('Provider failed'), ValueError('Bad plan'), []]:
            original = Mock()
            original.search.return_value = [{'id': 'original'}]
            sql = Mock()
            if isinstance(outcome, Exception):
                sql.search.side_effect = outcome
            else:
                sql.search.return_value = outcome
            router = chat_service.FallbackSqlRouter(sql, original)
            self.assertEqual(router.search('question'), [{'id': 'original'}])
            router.search('question')
            original.search.assert_called_once()
            self.assertTrue(router.used_fallback)

    def test_adapter_labels_match_and_diagnostics_do_not_block_answer(self):
        documents, original, sql, client = Mock(), Mock(), Mock(), Mock()
        documents.search.return_value = []
        original.expand_retrieval_query.return_value = 'expanded'
        sql.search.return_value = [{'id': 'structured:sql', 'text': 'Robert risk score 2', 'score': 1,
                                    'metadata': {'source_path': 'data/wealth.db', 'evidence_type': 'structured'}}]
        client.complete.return_value = 'Robert has risk score 2 [S1].'
        with patch.object(chat_service, 'resources', return_value=(documents, original)), patch.object(
            rag, 'SqlStructuredRouter', return_value=sql
        ), patch.object(rag.OpenAICompatibleChatClient, 'from_environment', return_value=client):
            result = chat_service.respond('Robert risk score?', structured_backend='sqlite')
        sql.search.assert_called_once()
        self.assertEqual(result['structured_backend'], 'sqlite')
        self.assertEqual(result['sources'][0]['label'], result['citations'][0]['label'])
        self.assertFalse(result['abstained'])
        self.assertEqual(result['grounding_diagnostics']['status'], 'unverified')

    def test_sql_preview_does_not_call_model(self):
        documents, original = Mock(), Mock()
        documents.search.return_value = []
        original.search.return_value = []
        with patch.object(chat_service, 'resources', return_value=(documents, original)), patch.object(
            rag.OpenAICompatibleChatClient, 'from_environment'
        ) as client:
            result = chat_service.respond('question', preview=True, structured_backend='sqlite')
        client.assert_not_called()
        self.assertEqual(result['structured_backend'], 'files')
