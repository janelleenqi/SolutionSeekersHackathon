# SolutionSeekersHackathon

Wealth Advisor Assistant prototype for evidence-grounded, citation-ready RAG over
the supplied APAC wealth-management dataset.

## Chat interface

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).
The UI reads `data/processed/chunks.jsonl`, supports session-only PDF, TXT, MD,
CSV and JSON uploads (20 MB per file), document-type filters, source expanders,
conversation reset, feedback and JSON conversation export. Uploaded documents
are processed only when you click **Process uploaded documents**; processing a
new batch replaces the previous uploaded batch. It does not modify the raw or
processed corpus. PDFs need extractable text; OCR is not included.

Until a backend is connected, the app explicitly runs in **Document search**
mode: local keyword ranking returns original evidence passages, or a no-match
message. This preview does not perform semantic retrieval, generate advice,
calculate portfolio exposure, or interpret follow-up questions using history.
It is a working UI, not the complete mandatory RAG implementation.

### Connect the RAG backend

Implement this function in `src/rag.py`; the UI detects it automatically:

```python
def answer(*, question, chunks, history, top_k):
    # Retrieve and generate using your backend here.
    # history contains earlier {"role": ..., "content": ...} messages.
    # Sources must be the actual passages supporting the answer.
    return {
        "answer": "Your grounded answer with citations such as [1].",
        "sources": [],  # Chunk dictionaries: id, text, metadata
        "abstained": False,
    }
```

The backend must respect the supplied filtered `chunks`, return sources in
citation order, and abstain when evidence is insufficient. It owns semantic
retrieval, model configuration and conversation-context handling. Source text
is displayed literally. Feedback is session-local and included in the download.
The interface uses Streamlit's [chat elements](https://docs.streamlit.io/develop/api-reference/chat).

## Person 1: ingestion and document processing

`src/ingestion.py` converts the raw corpus into vector-store-neutral
`data/processed/chunks.jsonl`. Each line contains a deterministic `id`, cleaned
`text`, and flat `metadata`. PDF metadata includes `source_path`, `page`, document
type, source/content SHA-256 hashes, and chunk index. Email messages also include
thread, message, client, date, sender, and recipient fields. This supports exact
passage citations and can be loaded directly into Chroma or paired with FAISS.

### Run

```powershell
python -m pip install -r requirements.txt
python -m src.ingestion --input data/raw --output data/processed --strict
```

The default corpus follows the dataset guidance: every PDF plus
`client_correspondence.json`. Portfolio and transaction files remain with the
structured-data component. Add record-level chunks for them when needed:

```powershell
python -m src.ingestion --include-structured
```

The run produces:

- `chunks.jsonl`: cleaned, overlapped, citation-ready passages.
- `manifest.json`: settings, per-source/type counts, and extraction errors.

To build the SQLite database represented by the relational schema, run the
database loader directly:

```powershell
python -m src.database --input data/raw --database data/wealth.db
```

Or build the document chunks and SQLite database together:

```powershell
python -m src.ingestion --input data/raw --output data/processed --database data/wealth.db --strict
```

The loader replaces the six structured tables in one transaction. It creates
`clients`, `products`, `holdings`, `transactions`, `email_threads`, and
`email_messages`, with foreign keys enabled and counts printed as JSON.

PDF chunks never cross page boundaries. Cleanup normalizes Unicode and whitespace,
repairs common line-break hyphenation, and conservatively removes repeated page
headers/footers. The process continues past a damaged file and logs it; `--strict`
makes any extraction error fail CI.

### Test

```powershell
python -m unittest discover -s tests -v
```

## Person 2: Sentence Transformer retrieval

The retrieval layer uses `sentence-transformers/all-MiniLM-L6-v2` to create
unit-normalized embeddings and stores them in a persistent Chroma collection using
cosine distance. The first run downloads the model; later runs use the local model
cache.

Build or update the index:

```powershell
python -m src.retrieval index
```

Retrieve citation-ready evidence:

```powershell
python -m src.retrieval search "Is the APEX Autocallable suitable for a Conservative client?" --top-k 5
```

Optional filters can be supplied with `--client-id CL002` or
`--document-type policy`. Each result includes its rank, cosine score, passage,
chunk ID, source path, and PDF page metadata.

Run the source-level retrieval evaluation:

```powershell
python -m src.retrieval evaluate --top-k 5
```

This reports Recall@K, Hit@K, and mean reciprocal rank over the cases in
`data/evaluation/retrieval_cases.json` and saves the full report to
`data/evaluation/retrieval_results.json`. Re-running `index` performs deterministic
upserts and removes stale chunk IDs, so unchanged chunks are not duplicated. The
generated `data/vector_db` directory is local and Git-ignored; rebuild it from the
tracked chunks after cloning the project.

## Person 4: grounded RAG answers

`src/rag.py` connects retrieval to an OpenAI-compatible chat endpoint. It labels
retrieved passages as `[S1]`, `[S2]`, and so on, requires those labels in the
answer, maps them back to real source paths/pages, and abstains when retrieval is
weak or the model produces missing or invented citations.

Configure the team's chosen compatible endpoint without committing secrets:

```powershell
$env:LLM_MODEL = "your-model-name"
$env:LLM_API_KEY = "your-api-key"
# Optional for a non-default compatible provider:
$env:LLM_BASE_URL = "https://provider.example/v1"
```

Preview exactly what evidence would be sent without calling an LLM:

```powershell
python -m src.rag "Is the APEX note suitable for a Conservative client?" --show-prompt
```

Generate a grounded answer:

```powershell
python -m src.rag "Is the APEX note suitable for a Conservative client?"
```

The JSON response includes the answer, an `abstained` flag, a machine-readable
reason, evidence count, and validated citations containing source path, page,
chunk ID, and retrieval score. API keys are read only from environment variables.

## Person 3: structured data and calculations

`src/structured_data.py` reads and validates the portfolio JSON, flattened holdings
CSV, transaction ledger, and client correspondence JSON. These records are not
stored in Chroma. Exact values, percentages, dates, transaction statuses,
comparisons, and email thread relationships are normalized in Python and returned
with source-file, record, and CSV-row references. The SQLite loader consumes this
validated structured view and stores all of these records in the relational tables.

Validate cross-file consistency:

```powershell
python -m src.structured_data validate
```

Example queries:

```powershell
python -m src.structured_data client "Robert Chua"
python -m src.structured_data exposure "Robert Chua" "APEX"
python -m src.structured_data transactions "James Sullivan" --status Pending
python -m src.structured_data risk-profile "Conservative"
python -m src.structured_data concentrations --threshold 20
```

At load time the module rejects duplicate identifiers, unknown client references,
invalid dates/numbers, portfolio allocations that do not total 100%, holding values
that do not reconcile to AUM, and differences between the JSON and flattened CSV.
It also validates correspondence thread IDs, client references, message fields, and
message dates before database insertion.
`StructuredResult.to_prompt_block()` converts a validated result into a labeled
evidence block. `rag.py` now performs that hybrid integration automatically:
named-client questions receive structured profile/holding/transaction evidence,
while Chroma supplies relevant policy, factsheet, and operational passages. For
suitability questions the structured product and risk profile also expand the
semantic search query, improving retrieval without changing the user's question.

```powershell
python -m src.rag "Is Robert Chua suitable for APEX?" --show-prompt
python -m src.rag "Is Robert Chua suitable for APEX?"
```

Use `--no-structured` to run document-only RAG. Structured citations include the
underlying JSON record or CSV row in `source_references`.
