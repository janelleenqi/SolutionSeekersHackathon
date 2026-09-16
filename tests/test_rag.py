import unittest

from src.rag import (
    ABSTENTION,
    RAGAssistant,
    StructuredQueryRouter,
    CoverageRetriever,
    build_evidence_prompt,
)
from src.structured_data import StructuredDataStore


EVIDENCE = [
    {
        "id": "policy-1",
        "text": "Complex products require a risk score of seven unless an exemption applies.",
        "score": 0.72,
        "metadata": {
            "source_path": "policy/suitability.pdf",
            "page": 2,
            "document_type": "policy",
        },
    },
    {
        "id": "fact-1",
        "text": "The product has an SRI rating of seven out of seven.",
        "score": 0.61,
        "metadata": {"source_path": "factsheets/apex.pdf", "page": 1},
    },
]


class FakeRetriever:
    def __init__(self, evidence=None):
        self.evidence = EVIDENCE if evidence is None else evidence
        self.calls = []

    def search(self, query, top_k=5, filters=None):
        self.calls.append((query, top_k, filters))
        return self.evidence[:top_k]


class FakeChatClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.response


class SequenceChatClient(FakeChatClient):
    def __init__(self, responses):
        super().__init__("")
        self.responses = iter(responses)

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return next(self.responses)


class RAGTests(unittest.TestCase):
    def test_documentation_review_corrects_requested_vs_missing(self):
        evidence = [{**EVIDENCE[0], "text": (
            "Compliance requested the trade file and RPQ. The RM replied: I cannot "
            "locate a signed product-specific acknowledgement; only a general disclosure exists."
        )}]
        client = SequenceChatClient([
            "The trade file and RPQ are missing [S1].",
            "The RM could not locate the product-specific acknowledgement. "
            "The trade file and RPQ were requested, not confirmed missing [S1].",
        ])
        result = RAGAssistant(FakeRetriever(evidence), client).answer("What documentation is missing?")
        self.assertFalse(result.abstained)
        self.assertIn("not confirmed missing", result.answer)
        self.assertIn("requested document", client.calls[1][1])

    def test_coverage_retrieval_respects_scope_and_budget(self):
        class Search:
            def __init__(self):
                self.calls = []
            def search(self, query, top_k=5, filters=None):
                self.calls.append(filters)
                return [{"id": str(len(self.calls)), "score": .8, "text": query}]
        delegate = Search()
        scope = {"document_type": {"$in": ["policy", "client_correspondence"]}}
        hits = CoverageRetriever(delegate).search("Is CL002 suitable? Documentation missing", 3, scope)
        self.assertEqual(len(hits), 3)
        self.assertEqual(delegate.calls[1]["$and"][0], scope)
        self.assertIn("CL002", str(delegate.calls[2]))

    def test_grounded_answer_maps_model_labels_to_real_sources(self):
        client = FakeChatClient("The product is high risk [S2] and requires risk matching [S1].")
        result = RAGAssistant(FakeRetriever(), client).answer("Is this suitable?")
        self.assertFalse(result.abstained)
        self.assertEqual([citation.label for citation in result.citations], ["S2", "S1"])
        self.assertEqual(result.citations[1].source_path, "policy/suitability.pdf")
        self.assertEqual(result.citations[1].page, 2)

    def test_low_relevance_abstains_before_calling_model(self):
        weak = [{**EVIDENCE[0], "score": 0.12}]
        client = FakeChatClient("Should not be used [S1].")
        result = RAGAssistant(FakeRetriever(weak), client).answer("Unrelated question")
        self.assertTrue(result.abstained)
        self.assertEqual(result.answer, ABSTENTION)
        self.assertEqual(result.reason, "low_retrieval_relevance")
        self.assertEqual(client.calls, [])

    def test_uncited_or_invented_citations_are_rejected(self):
        for response, reason in (
            ("An unsupported answer.", "answer_has_no_citations"),
            ("Invented evidence [S99].", "invalid_model_citations:S99"),
        ):
            with self.subTest(response=response):
                result = RAGAssistant(FakeRetriever(), FakeChatClient(response)).answer("Question")
                self.assertTrue(result.abstained)
                self.assertEqual(result.reason, reason)

    def test_uncited_answer_is_repaired_once(self):
        client = SequenceChatClient(
            [
                "The product is high risk.",
                "The product is high risk [S2].",
            ]
        )
        result = RAGAssistant(FakeRetriever(), client).answer("Is this high risk?")
        self.assertFalse(result.abstained)
        self.assertEqual(result.answer, "The product is high risk [S2].")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("failed citation validation", client.calls[1][1])

    def test_model_can_explicitly_abstain(self):
        result = RAGAssistant(
            FakeRetriever(), FakeChatClient("INSUFFICIENT_EVIDENCE")
        ).answer("Question")
        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, "model_found_insufficient_evidence")

    def test_prompt_contains_evidence_labels_and_provenance(self):
        prompt = build_evidence_prompt("Question?", EVIDENCE)
        self.assertIn("[S1]", prompt)
        self.assertIn("policy/suitability.pdf, page 2", prompt)
        self.assertIn("Chunk ID: policy-1", prompt)

    def test_structured_only_question_does_not_require_vector_evidence(self):
        router = StructuredQueryRouter(StructuredDataStore())
        client = FakeChatClient("Robert Chua has 15% in APEX, worth SGD 127,500 [S3].")
        result = RAGAssistant(
            FakeRetriever([]), client, structured_router=router
        ).answer("What is Robert Chua's APEX exposure?")
        self.assertFalse(result.abstained)
        self.assertEqual(result.evidence_count, 3)
        self.assertEqual(result.citations[0].evidence_type, "structured")
        self.assertEqual(
            result.citations[0].source_references[0]["source_path"],
            "client_portfolio/clients_portfolio.csv",
        )
        self.assertEqual(result.citations[0].source_references[0]["row"], 9)

    def test_hybrid_question_combines_structured_and_document_evidence(self):
        router = StructuredQueryRouter(StructuredDataStore())
        retriever = FakeRetriever()
        client = FakeChatClient(
            "Robert Chua has a Conservative profile [S1] and 15% APEX exposure [S3]. "
            "The product has an SRI rating of seven [S5]."
        )
        result = RAGAssistant(
            retriever, client, structured_router=router
        ).answer("Is Robert Chua suitable for APEX?")
        self.assertFalse(result.abstained)
        self.assertEqual(result.evidence_count, 5)
        self.assertEqual(
            [citation.evidence_type for citation in result.citations],
            ["structured", "structured", "document"],
        )
        prompt = client.calls[0][1]
        self.assertIn("Evidence type: structured", prompt)
        self.assertIn("Evidence type: document", prompt)
        self.assertIn(
            "APEX Global Multi-Asset Autocallable Note Series 7",
            retriever.calls[0][0],
        )
        self.assertIn("investment suitability policy", retriever.calls[0][0])


if __name__ == "__main__":
    unittest.main()
