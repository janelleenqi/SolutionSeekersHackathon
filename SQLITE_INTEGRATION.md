# Optional SQLite chat retrieval

The default remains the existing JSON/CSV structured router plus Chroma document
retrieval. Enable **Use SQLite structured queries** in the Streamlit sidebar to
try an LLM-planned, read-only SQL query instead. Turn it off to return to the
original path without reverting files.

The database must exist at `data/wealth.db`. Build or refresh it from the raw
dataset with `python -m src.database`. This command replaces the database's
dataset, so preserve any independently edited database before rebuilding.

SQLite mode costs an additional model request. Each question's structured
result is cached for that request so the displayed evidence and answer citations
use the same rows. Query expansion for Chroma still uses the original router.
SQL errors and empty results fall back to the original JSON/CSV queries, with a
visible notice. Evidence preview always uses the original local router and
never calls the model. A provider outage can still prevent answer generation.

SQL is read-only and bounded. Successful execution does not guarantee that the
model selected the right columns, rows or calculations; check against expected
answers before making this mode your default. Keep SQLite refreshed when source
data changes. Document ingestion and the Chroma index are unchanged.

Answer diagnostics report lexical overlap, absent numbers and unknown citation
labels. They do not verify factual entailment and do not suppress the answer.
An `unverified` diagnostic is not a finding that the answer is false.

Run regression tests with `python -m unittest discover -s tests -q`. SQL tests
exercise real SQLite with controlled model responses; they do not verify a live
provider's planning accuracy.
