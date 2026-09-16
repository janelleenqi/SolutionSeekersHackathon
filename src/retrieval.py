"""Sentence Transformer embeddings and persistent Chroma retrieval.

The ingestion layer writes vector-store-neutral JSONL.  This module embeds the
passages explicitly, stores them in Chroma, and returns citation metadata with
every result.  The model is loaded lazily so metric helpers and unit tests do not
download model weights.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "wealth_documents"
DEFAULT_DATABASE = Path("data/vector_db")
DEFAULT_CHUNKS = Path("data/processed/chunks.jsonl")
DEFAULT_EVALUATION = Path("data/evaluation/retrieval_cases.json")


class Encoder(Protocol):
    """Small interface that keeps the vector database testable."""

    model_name: str

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]:
        ...


class SentenceTransformerEncoder:
    """Lazily load a Sentence Transformer and produce unit-normalized vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence Transformer retrieval requires sentence-transformers; "
                    "run: python -m pip install -r requirements.txt"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load and validate the Person 1 JSONL handoff."""
    if not path.is_file():
        raise FileNotFoundError(f"Chunk file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            missing = {key for key in ("id", "text", "metadata") if key not in row}
            if missing:
                raise ValueError(f"Missing {sorted(missing)} on {path}:{line_number}")
            if not isinstance(row["metadata"], dict) or not row["text"].strip():
                raise ValueError(f"Invalid text or metadata on {path}:{line_number}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate chunk id {row['id']} on {path}:{line_number}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No chunks found in {path}")
    return rows


@dataclass(frozen=True)
class IndexSummary:
    collection: str
    embedding_model: str
    chunks_indexed: int
    stale_chunks_removed: int
    database_path: str


class VectorRetriever:
    """Persistent semantic index backed by Chroma cosine distance."""

    def __init__(
        self,
        database_path: Path = DEFAULT_DATABASE,
        collection_name: str = DEFAULT_COLLECTION,
        encoder: Encoder | None = None,
        client: Any | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Retrieval requires chromadb; install project requirements") from exc
        self.database_path = Path(database_path)
        self.collection_name = collection_name
        self.encoder = encoder or SentenceTransformerEncoder()
        self.client = client or chromadb.PersistentClient(path=str(self.database_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.encoder.model_name},
        )
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if self.collection.count() and stored_model != self.encoder.model_name:
            raise ValueError(
                f"Collection uses {stored_model!r}, but retrieval requested "
                f"{self.encoder.model_name!r}. Use a different collection or rebuild it."
            )

    def index(self, chunks_path: Path = DEFAULT_CHUNKS, batch_size: int = 32) -> IndexSummary:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        rows = load_chunks(Path(chunks_path))
        incoming_ids = {row["id"] for row in rows}
        stored_ids = set(self.collection.get(include=[])["ids"])
        stale_ids = sorted(stored_ids - incoming_ids)
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = self.encoder.encode(
                [row["text"] for row in batch], batch_size=batch_size
            )
            if len(vectors) != len(batch):
                raise ValueError("Embedding model returned the wrong number of vectors")
            self.collection.upsert(
                ids=[row["id"] for row in batch],
                documents=[row["text"] for row in batch],
                metadatas=[row["metadata"] for row in batch],
                embeddings=vectors,
            )
        return IndexSummary(
            collection=self.collection_name,
            embedding_model=self.encoder.model_name,
            chunks_indexed=len(rows),
            stale_chunks_removed=len(stale_ids),
            database_path=str(self.database_path.resolve()),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        count = self.collection.count()
        if count == 0:
            raise RuntimeError("Vector index is empty; run the index command first")
        query_vector = self.encoder.encode([query], batch_size=1)[0]
        # Over-fetch before removing overlapping chunks from the same citation
        # location. This stops one PDF page from consuming most of the evidence
        # window while still allowing separate pages from the same source.
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k * 3, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters
        response = self.collection.query(**kwargs)
        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]
        results: list[dict[str, Any]] = []
        seen_locations: set[tuple[Any, ...]] = set()
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            location = (
                metadata.get("source_path"),
                metadata.get("page"),
                metadata.get("record_id"),
            )
            if location in seen_locations:
                continue
            seen_locations.add(location)
            results.append(
                {
                    "rank": len(results) + 1,
                    "id": chunk_id,
                    "text": document,
                    "score": round(1.0 - float(distance), 6),
                    "distance": round(float(distance), 6),
                    "metadata": metadata,
                }
            )
            if len(results) == top_k:
                break
        return results


def retrieval_metrics(
    results: Sequence[dict[str, Any]], expected_sources: Sequence[str]
) -> dict[str, float]:
    """Calculate source-level recall, hit rate, and reciprocal rank."""
    expected = set(expected_sources)
    if not expected:
        raise ValueError("expected_sources cannot be empty")
    ranked_sources = [
        result.get("metadata", {}).get("source_path") for result in results
    ]
    found = expected.intersection(source for source in ranked_sources if source)
    first_relevant = next(
        (rank for rank, source in enumerate(ranked_sources, start=1) if source in expected), None
    )
    return {
        "recall": len(found) / len(expected),
        "hit": 1.0 if found else 0.0,
        "reciprocal_rank": 0.0 if first_relevant is None else 1.0 / first_relevant,
    }


def evaluate(
    retriever: VectorRetriever, cases_path: Path = DEFAULT_EVALUATION, top_k: int = 5
) -> dict[str, Any]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation file must contain a non-empty JSON list")
    case_results: list[dict[str, Any]] = []
    for case in cases:
        results = retriever.search(case["query"], top_k=top_k)
        metrics = retrieval_metrics(results, case["expected_sources"])
        case_results.append(
            {
                "id": case["id"],
                "query": case["query"],
                **metrics,
                "retrieved_sources": [result["metadata"].get("source_path") for result in results],
            }
        )
    return {
        "top_k": top_k,
        "case_count": len(case_results),
        "mean_recall_at_k": statistics.fmean(row["recall"] for row in case_results),
        "hit_rate_at_k": statistics.fmean(row["hit"] for row in case_results),
        "mean_reciprocal_rank": statistics.fmean(
            row["reciprocal_rank"] for row in case_results
        ),
        "cases": case_results,
    }


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", help="Sentence Transformer device, e.g. cpu or cuda")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentence Transformer semantic retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index", help="Embed and index chunks")
    _add_shared_arguments(index_parser)
    index_parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    index_parser.add_argument("--batch-size", type=int, default=32)

    search_parser = commands.add_parser("search", help="Retrieve top-K evidence")
    _add_shared_arguments(search_parser)
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--client-id")
    search_parser.add_argument("--document-type")

    evaluation_parser = commands.add_parser("evaluate", help="Measure source Recall@K")
    _add_shared_arguments(evaluation_parser)
    evaluation_parser.add_argument("--cases", type=Path, default=DEFAULT_EVALUATION)
    evaluation_parser.add_argument("--top-k", type=int, default=5)
    evaluation_parser.add_argument(
        "--output", type=Path, default=Path("data/evaluation/retrieval_results.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = SentenceTransformerEncoder(args.model, args.device)
    try:
        retriever = VectorRetriever(args.database, args.collection, encoder)
        if args.command == "index":
            output: Any = retriever.index(args.chunks, args.batch_size).__dict__
        elif args.command == "search":
            clauses = []
            if args.client_id:
                clauses.append({"client_id": args.client_id})
            if args.document_type:
                clauses.append({"document_type": args.document_type})
            filters = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}
            output = retriever.search(args.query, args.top_k, filters)
        else:
            output = evaluate(retriever, args.cases, args.top_k)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    except Exception as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
