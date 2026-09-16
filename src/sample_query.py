from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# model = SentenceTransformer(
#     "mukaj/fin-mpnet-base"
# )


# ============================================================
# 2. CONNECT TO EXISTING CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="wealth_documents"
)


# ============================================================
# 3. SEARCH FUNCTION
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

    return results


# ============================================================
# 4. TEST SEARCH
# ============================================================

if __name__ == "__main__":

    question = "conservative risk fund"

    results = search(question)

    print("Retrieved documents:\n")

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        print(f"--- Result {i + 1} ---")
        print(document)

        print(
            f"Source: {metadata['source_filename']}"
        )

        # Uncomment if page metadata is stored
        # print(f"Page: {metadata['page']}")

        print()