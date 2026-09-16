import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.retrieval import DEFAULT_MODEL


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "processed" / "query_logs.db"

EMBEDDING_MODEL = DEFAULT_MODEL
INDEX_VERSION = "v1"


# ============================================================
# DATABASE SETUP
# ============================================================

def get_connection() -> sqlite3.Connection:
    """Create a connection to the query log database."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            rank INTEGER NOT NULL,
            evidence_type TEXT,
            document_type TEXT,
            chunk_index INTEGER,
            source_filename TEXT,
            result TEXT,
            embedding_model TEXT,
            index_version TEXT
        )
    """)

    conn.commit()

    return conn


# ============================================================
# LOG RETRIEVED RESULTS
# ============================================================

def log_results(
    question: str,
    results: list[dict[str, Any]],
) -> None:
    """
    Store every retrieved evidence item in SQLite.

    One row = one retrieved result.
    """

    timestamp = datetime.now().isoformat(timespec="seconds")

    conn = get_connection()

    try:
        for rank, item in enumerate(results, start=1):
            metadata = item.get("metadata", {})
            document = item.get("text", "")

            conn.execute("""
                INSERT INTO query_logs (
                    timestamp,
                    query,
                    rank,
                    evidence_type,
                    document_type,
                    chunk_index,
                    source_filename,
                    result,
                    embedding_model,
                    index_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                question,
                rank,
                metadata.get("evidence_type"),
                metadata.get("document_type"),
                metadata.get("chunk_index"),
                metadata.get("source_filename"),
                document,
                EMBEDDING_MODEL,
                INDEX_VERSION,
            ))

        conn.commit()

    finally:
        conn.close()


# ============================================================
# VIEW QUERY LOGS
# ============================================================

def view_logs(
    timestamp: str,
    question: str,
) -> None:
    """
    Print all logged evidence for a specific query execution.
    """

    query = """
    SELECT *
    FROM query_logs
    WHERE timestamp = ?
    AND query = ?
    ORDER BY rank;
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(query, (timestamp, question))
        rows = cursor.fetchall()

        columns = [description[0] for description in cursor.description]

        print("=" * 100)
        print("QUERY LOG RESULTS")
        print("=" * 100)

        print(f"Database:  {DB_PATH.resolve()}")
        print(f"Timestamp: {timestamp}")
        print(f"Query:     {question}")
        print(f"Results:   {len(rows)}")
        print()

        for row in rows:
            print("-" * 100)

            for column, value in zip(columns, row):
                print(f"{column}: {value}")

        print("=" * 100)

    finally:
        conn.close()


# ============================================================
# DATABASE PATH
# ============================================================

def get_db_path() -> Path:
    """Return the query log database path."""
    return DB_PATH


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    timestamp = "2026-09-16T16:20:01"
    question = "Which fund has a conservative risk profile?"

    view_logs(timestamp, question)