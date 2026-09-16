from sentence_transformers import SentenceTransformer
import chromadb
import re


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="wealth_documents"
)


# ============================================================
# 3. SEMANTIC SEARCH
# ============================================================

def document_search(question, top_k=5):
    """
    Semantic search.

    Returns the top_k most semantically relevant chunks.
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
# 4. STRUCTURED SEARCH
# ============================================================

def structured_search(
    client_id=None,
    document_type=None,
    date=None
):
    """
    Exact metadata-based search.

    Returns ALL records matching the filters.
    No top_k is applied.
    """

    filters = []

    if client_id:
        filters.append({
            "client_id": client_id
        })

    if document_type:
        filters.append({
            "document_type": document_type
        })

    if date:
        filters.append({
            "date": date
        })

    # No filters
    if len(filters) == 0:

        results = collection.get()

    # One filter
    elif len(filters) == 1:

        results = collection.get(
            where=filters[0]
        )

    # Multiple filters
    else:

        results = collection.get(
            where={
                "$and": filters
            }
        )

    return results


# ============================================================
# 5. DETERMINE WHETHER QUESTION IS STRUCTURED
# ============================================================

def parse_structured_query(question):

    question_lower = question.lower()

    filters = {}

    # --------------------------------------------------------
    # CLIENT ID
    # --------------------------------------------------------

    client_match = re.search(
        r"\bCL\d+\b",
        question,
        re.IGNORECASE
    )

    if client_match:
        filters["client_id"] = client_match.group(0).upper()

    # --------------------------------------------------------
    # DOCUMENT TYPE
    # --------------------------------------------------------

    if "correspondence" in question_lower:
        filters["document_type"] = "client_correspondence"

    elif "factsheet" in question_lower:
        filters["document_type"] = "fund_factsheet"

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    date_match = re.search(
        r"\b\d{4}-\d{2}-\d{2}\b",
        question
    )

    if date_match:
        filters["date"] = date_match.group(0)

    # --------------------------------------------------------
    # STRUCTURED QUESTION KEYWORDS
    # --------------------------------------------------------

    structured_keywords = [
        "client id",
        "document type",
        "record id",
        "date",
        "sender",
        "recipient",
        "subject",
        "thread id",
        "conservative risk profile",
        "expense ratio",
        "aum",
        "fund size",
        "currency"
    ]

    has_structured_keyword = any(
        keyword in question_lower
        for keyword in structured_keywords
    )

    # Actual filter found
    if filters:
        return filters

    # Structured question but no exact filter extracted
    if has_structured_keyword:
        return {}

    # Otherwise semantic search
    return None


# ============================================================
# 6. SMART SEARCH
# ============================================================

def search(question, top_k=5):

    structured_filters = parse_structured_query(question)

    # --------------------------------------------------------
    # STRUCTURED SEARCH
    # --------------------------------------------------------

    if structured_filters is not None:

        print("Search type: STRUCTURED")
        print("Filters:", structured_filters)

        # IMPORTANT:
        # No top_k here.
        # Return ALL valid matching records.

        results = structured_search(
            **structured_filters
        )

        return {
            "type": "structured",
            "results": results[:1] # 1st for testing purposes
        }

    # --------------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------------

    print("Search type: SEMANTIC / RAG")
    print("Top K:", top_k)

    # Only semantic search uses top_k
    results = document_search(
        question,
        top_k=top_k
    )

    return {
        "type": "semantic",
        "results": results
    }


# ============================================================
# 7. PRINT RESULTS
# ============================================================

def print_results(results):

    print("\n" + "=" * 60)

    # ========================================================
    # STRUCTURED RESULTS
    # ========================================================

    if results["type"] == "structured":

        data = results["results"]

        print("STRUCTURED RESULTS")
        print("=" * 60)

        if not data["ids"]:
            print("No matching records found.")
            return

        print(f"Found {len(data['ids'])} matching records.")

        for i in range(len(data["ids"])):

            print(f"\n--- Result {i + 1} ---")

            print("ID:")
            print(data["ids"][i])

            print("\nMetadata:")
            print(data["metadatas"][i])

            print("\nText:")
            print(data["documents"][i])

    # ========================================================
    # SEMANTIC RESULTS
    # ========================================================

    elif results["type"] == "semantic":

        data = results["results"]

        print("SEMANTIC / RAG RESULTS")
        print("=" * 60)

        for i, document in enumerate(
            data["documents"][0]
        ):

            metadata = data["metadatas"][0][i]

            print(f"\n--- Result {i + 1} ---")

            print("Document:")
            print(document)

            print("\nMetadata:")
            print(metadata)

            print(
                "\nDistance:",
                data["distances"][0][i]
            )


# ============================================================
# 8. TEST
# ============================================================

if __name__ == "__main__":

    question = "Which fund has a conservative risk profile?"

    results = search(
        question,
        top_k=5
    )

    print_results(results)