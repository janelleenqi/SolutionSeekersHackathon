"""Adapter connecting Streamlit to the CLI's hybrid RAG pipeline."""
from functools import lru_cache
from pathlib import Path
import sqlite3
from src import rag
from src.retrieval import load_chunks, VectorRetriever, SentenceTransformerEncoder
from src.structured_data import StructuredDataStore
from src.hallucination import check_hallucination

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

class FallbackSqlRouter:
    """Request-local cache keeps prompt evidence and displayed citations identical."""
    def __init__(self, sql_router, original):
        self.sql_router = sql_router
        self.original = original
        self.cache = {}
        self.used_fallback = False

    def search(self, question):
        if question not in self.cache:
            try:
                result = self.sql_router.search(question)
            except (ValueError, OSError, RuntimeError, sqlite3.DatabaseError):
                result = []
            if not result:
                self.used_fallback = True
                result = self.original.search(question)
            self.cache[question] = result
        return self.cache[question]

    def expand_retrieval_query(self, question):
        return self.original.expand_retrieval_query(question)


def respond(question, top_k=5, document_types=None, preview=False, structured_backend='files'):
    if structured_backend not in {'files', 'sqlite'}:
        raise ValueError('structured_backend must be files or sqlite')
    retriever, router = resources()
    # Preview remains entirely local: SQL planning requires an LLM call.
    if structured_backend == 'sqlite' and not preview:
        router = FallbackSqlRouter(rag.SqlStructuredRouter(
            rag.OpenAICompatibleChatClient.from_environment(), ROOT / 'data/wealth.db'), router)
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
    result['structured_backend'] = ('sqlite' if isinstance(router, FallbackSqlRouter)
                                    and not router.used_fallback else 'files')
    result['structured_fallback'] = isinstance(router, FallbackSqlRouter) and router.used_fallback
    if not preview:
        result['grounding_diagnostics'] = check_hallucination(result['answer'], result['sources'])
    return result
