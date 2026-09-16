"""Adapter connecting Streamlit to the CLI's hybrid RAG pipeline."""
from functools import lru_cache
from pathlib import Path
from src import rag
from src.retrieval import load_chunks, VectorRetriever, SentenceTransformerEncoder
from src.structured_data import StructuredDataStore

ROOT = Path(__file__).resolve().parents[1]

def backend_ready():
    try:
        rag.OpenAICompatibleChatClient.from_environment()
        return True
    except ValueError:
        return False

@lru_cache(maxsize=1)
def resources():
    return (rag.CoverageRetriever(VectorRetriever(ROOT / 'data/vector_db', encoder=SentenceTransformerEncoder())),
            rag.StructuredQueryRouter(StructuredDataStore(ROOT / 'data/raw/client_portfolio')))

class CapturedRetriever:
    def __init__(self, delegate):
        self.delegate = delegate
        self.evidence = []
    def search(self, query, top_k=5, filters=None):
        self.evidence = self.delegate.search(query, top_k=top_k, filters=filters)
        return self.evidence

def respond(question, top_k=5, document_types=None, preview=False):
    retriever, router = resources()
    captured = CapturedRetriever(retriever)
    filters = None
    if document_types is not None:
        if not document_types:
            raise ValueError('Select at least one document type.')
        filters = {'document_type': {'$in': document_types}}
    structured = router.search(question)
    if preview:
        documents = captured.search(router.expand_retrieval_query(question), top_k, filters)
        result = {'answer': 'Review the retrieved evidence below. No LLM answer was generated.',
                  'abstained': False, 'reason': None, 'citations': []}
    else:
        client = rag.OpenAICompatibleChatClient.from_environment()
        result = rag.RAGAssistant(captured, client, structured_router=router).answer(question, top_k, filters).to_dict()
        documents = captured.evidence
    if structured:
        documents = [item for item in documents if float(item.get('score', 0)) >= 0.30]
    evidence = structured + documents
    cited = {c['label'] for c in result['citations']}
    result.update(sources=[{**item, 'label': f'S{index}', 'cited': f'S{index}' in cited}
                           for index, item in enumerate(evidence, 1)],
                  evidence_count=len(evidence), mode='Evidence preview' if preview else 'Hybrid RAG')
    return result
