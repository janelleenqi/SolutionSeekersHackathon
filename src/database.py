"""Load the structured wealth-management source files into SQLite."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from src.structured_data import StructuredDataStore

DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_DATABASE = Path("data/wealth.db")

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS clients (
    client_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    nationality TEXT,
    residency_country TEXT,
    age INTEGER,
    occupation TEXT,
    marital_status TEXT,
    net_worth_band TEXT,
    investor_status TEXT,
    base_currency TEXT,
    aum REAL NOT NULL,
    risk_profile TEXT,
    risk_score INTEGER CHECK (risk_score BETWEEN 1 AND 10),
    investment_objective TEXT,
    relationship_manager TEXT,
    servicing_branch TEXT,
    kyc_status TEXT,
    pep_status TEXT,
    source_of_wealth TEXT,
    last_portfolio_review_date TEXT,
    suitability_flag INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_name TEXT PRIMARY KEY,
    asset_class TEXT
);

CREATE TABLE IF NOT EXISTS holdings (
    holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL REFERENCES clients(client_id),
    allocation_percentage REAL NOT NULL CHECK (allocation_percentage BETWEEN 0 AND 100),
    product_name TEXT NOT NULL REFERENCES products(product_name),
    value_sgd REAL NOT NULL CHECK (value_sgd >= 0),
    currency TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients(client_id),
    transaction_date TEXT NOT NULL,
    transaction_type TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount >= 0),
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT,
    product_name TEXT REFERENCES products(product_name)
);

CREATE TABLE IF NOT EXISTS email_threads (
    thread_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients(client_id),
    subject TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS email_messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES email_threads(thread_id),
    message_order INTEGER NOT NULL,
    sender_email TEXT NOT NULL,
    recipient_email TEXT NOT NULL,
    sent_date TEXT NOT NULL,
    body TEXT NOT NULL,
    UNIQUE(thread_id, message_order)
);
"""


def _required_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file does not exist: {path}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with _required_file(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> Any:
    return json.loads(_required_file(path).read_text(encoding="utf-8-sig"))


def _as_bool(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value.casefold() in {"1", "true", "yes"})


def _client_rows(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    fields = (
        "client_id", "name", "nationality", "residency_country", "age", "occupation",
        "marital_status", "net_worth_band", "investor_status", "base_currency", "aum_sgd",
        "risk_profile", "risk_score_1_to_10", "investment_objective", "relationship_manager",
        "servicing_branch", "kyc_status", "pep_status", "source_of_wealth",
        "last_portfolio_review_date", "suitability_flag", "notes",
    )
    return [
        tuple(client.get(field) for field in fields[:-2])
        + (_as_bool(client.get("suitability_flag")), client.get("notes"))
        for client in payload.get("clients", [])
    ]


def load_database(input_dir: Path = DEFAULT_INPUT_DIR, database_path: Path = DEFAULT_DATABASE) -> dict[str, int]:
    """Replace the SQLite dataset with records from the raw source files."""
    input_dir = Path(input_dir)
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    structured = StructuredDataStore(input_dir / "client_portfolio")
    clients_payload = {"clients": structured.clients}
    holdings = structured.holdings
    transactions = structured.transactions

    product_classes: dict[str, str | None] = {}
    for row in holdings:
        product_classes.setdefault(row["product_name"], row.get("asset_class"))
    for row in transactions:
        product_classes.setdefault(row["product_name"], None)

    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(SCHEMA)
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.execute("DELETE FROM email_messages")
            connection.execute("DELETE FROM email_threads")
            connection.execute("DELETE FROM holdings")
            connection.execute("DELETE FROM transactions")
            connection.execute("DELETE FROM products")
            connection.execute("DELETE FROM clients")
            connection.executemany(
                "INSERT INTO clients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                _client_rows(clients_payload),
            )
            connection.executemany(
                "INSERT INTO products(product_name, asset_class) VALUES (?, ?)",
                product_classes.items(),
            )
            connection.executemany(
                """INSERT INTO holdings
                (client_id, allocation_percentage, product_name, value_sgd, currency)
                VALUES (?, ?, ?, ?, ?)""",
                [
                    (row["client_id"], float(row["allocation_pct"]), row["product_name"],
                     float(row["value_sgd"]), row["holding_currency"])
                    for row in holdings
                ],
            )
            connection.executemany(
                """INSERT INTO transactions
                (transaction_id, client_id, transaction_date, transaction_type, product_name,
                 amount, currency, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (row["transaction_id"], row["client_id"], row["date"], row["transaction_type"],
                     row["product_name"], float(row["amount"]), row["currency"], row["status"],
                     row.get("notes") or None)
                    for row in transactions
                ],
            )
            for thread in structured.email_threads:
                connection.execute(
                    "INSERT INTO email_threads(thread_id, client_id, subject) VALUES (?, ?, ?)",
                    (thread["thread_id"], thread["client_id"], thread["subject"]),
                )
            connection.executemany(
                """INSERT INTO email_messages
                (thread_id, message_order, sender_email, recipient_email, sent_date, body)
                VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (message["thread_id"], message["message_order"], message["sender_email"],
                     message["recipient_email"], message["sent_date"], message["body"])
                    for message in structured.email_messages
                ],
            )
    finally:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("clients", "products", "holdings", "transactions", "email_threads", "email_messages")
        }
        connection.close()
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load raw wealth data into SQLite")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(load_database(args.input, args.database), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())