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
