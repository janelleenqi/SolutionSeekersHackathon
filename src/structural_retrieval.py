import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "wealth.db"


def retrieve_structured_data(sql_query, params=(), *, database_path=None, as_dict=False,
                             max_rows=1000):
    """
    Execute a bounded, read-only query. Existing callers still receive tuples.
    Set as_dict=True for named columns. This is not a RAG evidence adapter.
    Missing databases are never silently created; build with src.database first.
    """

    if not isinstance(sql_query, str) or not sql_query.strip():
        raise ValueError("sql_query must be a nonempty string")
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    path = Path(database_path if database_path is not None else DB_PATH).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}. Run python -m src.database first.")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                   sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() == "load_extension":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        conn.set_authorizer(authorize)
        budget = 1000
        def progress():
            nonlocal budget
            budget -= 1
            return int(budget <= 0)
        conn.set_progress_handler(progress, 1000)
        cursor = conn.execute(sql_query, params)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError(f"Query exceeds max_rows={max_rows}; narrow the query.")
        if as_dict:
            columns = [column[0] for column in cursor.description]
            if len(set(columns)) != len(columns):
                raise ValueError("Duplicate column names; use SQL aliases for named results.")
            return [dict(zip(columns, row)) for row in rows]
        return rows
    finally:
        conn.close()
