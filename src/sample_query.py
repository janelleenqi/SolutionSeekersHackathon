from sentence_transformers import SentenceTransformer
import chromadb
import sqlite3
from datetime import datetime


# ============================================================
# 1. CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "mukaj/fin-mpnet-base"

INDEX_VERSION = "v1"


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# 3. CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="wealth_documents"
)


# ============================================================
# 4. CONNECT TO SQLITE
# ============================================================

db_path = "../data/processed/query_logs.db"
db = sqlite3.connect(db_path)

cursor = db.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        query TEXT NOT NULL,
        rank INTEGER NOT NULL,
        document_type TEXT,
        chunk_index INTEGER,
        source_filename TEXT,
        result TEXT,
        embedding_model TEXT,
        index_version TEXT
    )
""")

db.commit()


# ============================================================
# 5. LOG RETRIEVED RESULTS
# ============================================================

def log_results(question, results):
    """
    Store every retrieved document chunk in SQLite.
    One row = one retrieved result.
    """

    timestamp = datetime.now().isoformat(timespec="seconds")

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i, document in enumerate(documents):

        metadata = metadatas[i]

        cursor.execute("""
            INSERT INTO query_logs (
                timestamp,
                query,
                rank,
                document_type,
                chunk_index,
                source_filename,
                result,
                embedding_model,
                index_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            question,
            i + 1,
            metadata.get("document_type"),
            metadata.get("chunk_index"),
            metadata.get("source_filename"),
            document,
            EMBEDDING_MODEL,
            INDEX_VERSION
        ))

    db.commit()


# ============================================================
# 6. SEARCH FUNCTION
# ============================================================

def search(question, top_k=5):
    """
    Search the vector database for chunks
    relevant to the question.
    """

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # Log retrieved results
    log_results(
        question,
        results
    )

    return results


# ============================================================
# 7. TEST SEARCH
# ============================================================

if __name__ == "__main__":

    question = "conservative risk fund"

    results = search(
        question,
        top_k=5
    )

    print("Retrieved documents:\n")

    for i, document in enumerate(
        results["documents"][0]
    ):

        metadata = results["metadatas"][0][i]

        print(f"--- Result {i + 1} ---")

        print(document)

        print(
            f"Source: {metadata.get('source_filename')}"
        )

        print(
            f"Document type: {metadata.get('document_type')}"
        )

        print(
            f"Chunk index: {metadata.get('chunk_index')}"
        )

        print()