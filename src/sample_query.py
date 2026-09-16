from sentence_transformers import SentenceTransformer
import chromadb


# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#model = SentenceTransformer("mukaj/fin-mpnet-base")


# Connect to existing ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="wealth_documents"
)


def search(question, top_k=5):
    """
    Search the vector database for chunks relevant to the question.
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


if __name__ == "__main__":
    question = "What risk fund are more conservative?"

    results = search(question)

    print("Retrieved documents:\n")

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]

        print(f"--- Result {i + 1} ---")
        print(document)
        print(f"Source: {metadata['source_filename']}")
        #print(f"Page: {metadata['page']}")
        print()