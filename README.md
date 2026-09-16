# SolutionSeekersHackathon

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
