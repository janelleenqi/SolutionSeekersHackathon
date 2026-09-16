import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#model = SentenceTransformer("mukaj/fin-mpnet-base")

# Load your chunks
with open("../data/processed/chunks.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Chroma
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("wealth_documents")
    print("Deleted existing wealth_documents collection")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="wealth_documents"
)

# Prepare data
ids = []
documents = []
metadatas = []

for chunk in data:
    ids.append(chunk["id"])
    documents.append(chunk["text"])

    metadata = chunk["metadata"]

    # Keep only useful retrieval/citation metadata
    # metadatas.append({
    #     "document_type": metadata["document_type"],
    #     #"page": metadata["page"],
    #     "source_filename": metadata["source_filename"],
    #     #"title": metadata["title"],
    #     "chunk_index": metadata["chunk_index"]
    # })

    metadatas.append({
        "document_type": metadata.get("document_type", ""),
        "source_filename": metadata.get("source_filename", ""),
        "chunk_index": metadata.get("chunk_index", 0)
    })

# Generate embeddings

embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
).tolist()

# Store in Chroma
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)