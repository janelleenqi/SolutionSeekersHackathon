
import sqlite3
from pathlib import Path

# ============================================================
# 1. DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / ".." /"data" / "processed" / "query_logs.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. QUERY
# ============================================================

timestamp = "2026-09-16T16:20:01"
question = "Which fund has a conservative risk profile?"

query = """
SELECT *
FROM query_logs
WHERE timestamp = ?
AND query = ?
ORDER BY rank;
"""


# ============================================================
# 3. RUN QUERY
# ============================================================

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute(query, (timestamp, question))
rows = cursor.fetchall()


# ============================================================
# 4. PRINT RESULTS
# ============================================================

columns = [description[0] for description in cursor.description]

print("=" * 100)
print("QUERY LOG RESULTS")
print("=" * 100)

print(f"Timestamp: {timestamp}")
print(f"Query:    {question}")
print(f"Results:  {len(rows)}")
print()

for row in rows:
    print("-" * 100)

    for column, value in zip(columns, row):
        print(f"{column}: {value}")

print("=" * 100)

conn.close()
