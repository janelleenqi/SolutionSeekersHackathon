# SolutionSeekersHackathon

## Frontend

Start from the same PowerShell terminal where your LLM variables are configured:

```powershell
python -m streamlit run app.py
```

The UI also reads a local `.env` file (existing environment variables take priority).
Set `LLM_MODEL`, `LLM_API_KEY`, and `LLM_BASE_URL` for your provider.
The frontend uses the existing Chroma index and validated structured data through
`src/chat_service.py`. Evidence preview works without LLM credentials. Each question
is independent; include the client name or ID. Source expanders retain the exact
answer citation labels and show retrieved passages and structured record references.
Document-type filters affect only document retrieval. Feedback and exported chat
history are session-local. Uploads are not enabled; add documents through ingestion
and rebuild the index before restarting the frontend.

Wealth Advisor Assistant prototype for evidence-grounded, citation-ready RAG over
the supplied APAC wealth-management dataset.

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

## Person 4: grounded RAG answers

Suitability/documentation queries reserve document slots for targeted policy and
client-correspondence searches while retaining the selected Top-K limit and filters.
Documentation questions receive one additional LLM review to distinguish records
requested by Compliance from records explicitly reported missing. This adds latency
and token usage; citation-label validation and an LLM review do not guarantee factual
correctness. Continue evaluating the final answers against the golden dataset.

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

`src/structured_data.py` reads the portfolio JSON, flattened holdings CSV, and
transaction ledger directly. These records are not stored in Chroma. Exact values,
percentages, dates, transaction statuses, and comparisons are calculated in Python
and returned with source-file, record, and CSV-row references.

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
