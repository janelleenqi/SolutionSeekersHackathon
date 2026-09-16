import unittest
from unittest.mock import patch
from src import chat_service
from src.rag import StructuredQueryRouter
from src.structured_data import StructuredDataStore


class Documents:
    def search(self, query, top_k=5, filters=None):
        self.filters = filters
        return [{'id': 'policy', 'text': 'Policy evidence', 'score': .8,
                 'metadata': {'source_path': 'policy.pdf', 'page': 1}}]


class Chat:
    def complete(self, system_prompt, user_prompt):
        return 'Robert has a Conservative profile [S1]. Policy evidence [S4].'


class ChatServiceTests(unittest.TestCase):
    def test_hybrid_labels_match_sources_and_filters(self):
        documents = Documents()
        router = StructuredQueryRouter(StructuredDataStore())
        with patch.object(chat_service, 'resources', return_value=(documents, router)), patch.object(
            chat_service.rag.OpenAICompatibleChatClient, 'from_environment', return_value=Chat()
        ):
            result = chat_service.respond('Is Robert Chua suitable for APEX?', document_types=['policy'])
        self.assertFalse(result['abstained'])
        self.assertEqual([s['label'] for s in result['sources'] if s['cited']], ['S1', 'S4'])
        self.assertEqual(result['sources'][3]['id'], 'policy')
        self.assertEqual(documents.filters, {'document_type': {'$in': ['policy']}})

    def test_preview_never_calls_llm(self):
        router = StructuredQueryRouter(StructuredDataStore())
        with patch.object(chat_service, 'resources', return_value=(Documents(), router)), patch.object(
            chat_service.rag.OpenAICompatibleChatClient, 'from_environment'
        ) as client:
            result = chat_service.respond('What pending transactions does James Sullivan have?', preview=True)
        client.assert_not_called()
        self.assertIn('TXN-1047', str(result['sources']))
        self.assertEqual(result['mode'], 'Evidence preview')
