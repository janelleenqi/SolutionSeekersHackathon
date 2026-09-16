"""Deterministic queries and calculations over client and transaction data.

Structured portfolio facts remain separate from semantic retrieval.  This module
loads the authoritative JSON/CSV records, validates their internal consistency,
and returns calculation results together with source references suitable for a
hybrid RAG prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

DEFAULT_DATA_DIR = Path("data/raw/client_portfolio")
CLIENT_JSON = "clients_portfolio.json"
HOLDINGS_CSV = "clients_portfolio.csv"
TRANSACTIONS_CSV = "transactions.csv"
COMPLEX_ASSET_KEYWORDS = ("complex", "structured", "specified investment", "sip")


class StructuredDataError(ValueError):
    """Raised when authoritative structured records are missing or inconsistent."""


@dataclass(frozen=True)
class SourceReference:
    source_path: str
    record_id: str
    row: int | None = None


@dataclass(frozen=True)
class StructuredResult:
    operation: str
    summary: str
    data: Any
    sources: list[SourceReference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "summary": self.summary,
            "data": self.data,
            "sources": [asdict(source) for source in self.sources],
        }

    def to_prompt_block(self, label: str = "D1") -> str:
        """Format a trusted calculation for later inclusion in a RAG prompt."""
        source_text = "; ".join(
            f"{source.source_path}"
            + (f", row {source.row}" if source.row is not None else "")
            + f", record {source.record_id}"
            for source in self.sources
        )
        return (
            f"[{label}]\nType: structured calculation\nOperation: {self.operation}\n"
            f"Summary: {self.summary}\nSources: {source_text}\n"
            f"Data: {json.dumps(self.data, ensure_ascii=False, sort_keys=True)}"
        )


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _parse_int(value: str, field: str, source: str, row: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise StructuredDataError(f"{source} row {row}: {field} must be an integer") from exc


def _parse_float(value: str, field: str, source: str, row: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StructuredDataError(f"{source} row {row}: {field} must be numeric") from exc
    if not math.isfinite(number):
        raise StructuredDataError(f"{source} row {row}: {field} must be finite")
    return number


def _clean_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


class StructuredDataStore:
    """Validated in-memory view of client, holding, and transaction records."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.clients = self._load_clients()
        self.holdings = self._load_holdings()
        self.transactions = self._load_transactions()
        self.clients_by_id = {client["client_id"]: client for client in self.clients}
        self.holdings_by_client: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.transactions_by_client: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for holding in self.holdings:
            self.holdings_by_client[holding["client_id"]].append(holding)
        for transaction in self.transactions:
            self.transactions_by_client[transaction["client_id"]].append(transaction)
        self.validation = self._validate()

    @staticmethod
    def _source_path(filename: str) -> str:
        return f"client_portfolio/{filename}"

    def _require_file(self, filename: str) -> Path:
        path = self.data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Structured data file does not exist: {path}")
        return path

    def _load_clients(self) -> list[dict[str, Any]]:
        path = self._require_file(CLIENT_JSON)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        clients = payload.get("clients") if isinstance(payload, dict) else None
        if not isinstance(clients, list) or not clients:
            raise StructuredDataError(f"{CLIENT_JSON} must contain a non-empty clients list")
        return clients

    def _load_holdings(self) -> list[dict[str, Any]]:
        path = self._require_file(HOLDINGS_CSV)
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, raw in enumerate(csv.DictReader(handle), start=2):
                row = dict(raw)
                row["age"] = _parse_int(row["age"], "age", HOLDINGS_CSV, row_number)
                row["risk_score_1_to_10"] = _parse_int(
                    row["risk_score_1_to_10"], "risk_score_1_to_10", HOLDINGS_CSV, row_number
                )
                for field in ("aum_sgd", "allocation_pct", "value_sgd"):
                    row[field] = _parse_float(row[field], field, HOLDINGS_CSV, row_number)
                row["_source_row"] = row_number
                rows.append(row)
        if not rows:
            raise StructuredDataError(f"{HOLDINGS_CSV} contains no records")
        return rows

    def _load_transactions(self) -> list[dict[str, Any]]:
        path = self._require_file(TRANSACTIONS_CSV)
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, raw in enumerate(csv.DictReader(handle), start=2):
                row = dict(raw)
                try:
                    date.fromisoformat(row["date"])
                except (TypeError, ValueError) as exc:
                    raise StructuredDataError(
                        f"{TRANSACTIONS_CSV} row {row_number}: date must use YYYY-MM-DD"
                    ) from exc
                row["amount"] = _parse_float(
                    row["amount"], "amount", TRANSACTIONS_CSV, row_number
                )
                row["_source_row"] = row_number
                rows.append(row)
        if not rows:
            raise StructuredDataError(f"{TRANSACTIONS_CSV} contains no records")
        return rows

    def _validate(self) -> dict[str, Any]:
        errors: list[str] = []
        client_ids = [str(client.get("client_id", "")) for client in self.clients]
        duplicates = sorted(key for key, count in Counter(client_ids).items() if count > 1)
        if duplicates:
            errors.append(f"duplicate client IDs: {duplicates}")
        known_ids = set(client_ids)
        holding_ids = {holding["client_id"] for holding in self.holdings}
        transaction_ids = {transaction["client_id"] for transaction in self.transactions}
        if holding_ids != known_ids:
            errors.append(
                f"holding/client ID mismatch: missing={sorted(known_ids-holding_ids)}, "
                f"unknown={sorted(holding_ids-known_ids)}"
            )
        unknown_transaction_clients = sorted(transaction_ids - known_ids)
        if unknown_transaction_clients:
            errors.append(f"transactions reference unknown clients: {unknown_transaction_clients}")

        duplicate_transactions = sorted(
            key
            for key, count in Counter(t["transaction_id"] for t in self.transactions).items()
            if count > 1
        )
        if duplicate_transactions:
            errors.append(f"duplicate transaction IDs: {duplicate_transactions}")

        json_clients = {client["client_id"]: client for client in self.clients}
        for client_id in sorted(known_ids):
            holdings = [row for row in self.holdings if row["client_id"] == client_id]
            allocation = sum(row["allocation_pct"] for row in holdings)
            value = sum(row["value_sgd"] for row in holdings)
            client = json_clients[client_id]
            if not math.isclose(allocation, 100.0, abs_tol=0.01):
                errors.append(f"{client_id} allocations total {allocation}, expected 100")
            if not math.isclose(value, float(client["aum_sgd"]), abs_tol=0.01):
                errors.append(
                    f"{client_id} holding values total {value}, expected AUM {client['aum_sgd']}"
                )
            json_holdings = {
                holding["product_name"]: holding for holding in client.get("portfolio_holdings", [])
            }
            csv_names = {holding["product_name"] for holding in holdings}
            if set(json_holdings) != csv_names:
                errors.append(f"{client_id} JSON/CSV product names do not reconcile")
            for holding in holdings:
                original = json_holdings.get(holding["product_name"])
                if original and (
                    not math.isclose(
                        holding["allocation_pct"], float(original["allocation_pct"]), abs_tol=0.01
                    )
                    or not math.isclose(
                        holding["value_sgd"], float(original["value_sgd"]), abs_tol=0.01
                    )
                ):
                    errors.append(
                        f"{client_id} {holding['product_name']} differs between JSON and CSV"
                    )
            first = holdings[0]
            for field in ("name", "risk_profile"):
                if first[field] != client[field]:
                    errors.append(f"{client_id} field {field} differs between JSON and CSV")
            if first["risk_score_1_to_10"] != int(client["risk_score_1_to_10"]):
                errors.append(
                    f"{client_id} field risk_score_1_to_10 differs between JSON and CSV"
                )
            if not math.isclose(
                first["aum_sgd"], float(client["aum_sgd"]), abs_tol=0.01
            ):
                errors.append(f"{client_id} field aum_sgd differs between JSON and CSV")

        if any(row["amount"] < 0 for row in self.transactions):
            errors.append("transaction amounts must not be negative")
        if errors:
            raise StructuredDataError("Structured data validation failed: " + "; ".join(errors))
        return {
            "valid": True,
            "client_count": len(self.clients),
            "holding_count": len(self.holdings),
            "transaction_count": len(self.transactions),
            "pending_transaction_count": sum(
                transaction["status"].casefold() == "pending"
                for transaction in self.transactions
            ),
            "checks": [
                "unique client and transaction IDs",
                "known client IDs in holdings and transactions",
                "portfolio allocations reconcile to 100%",
                "holding values reconcile to client AUM",
                "JSON holdings reconcile to flattened CSV holdings",
                "ISO transaction dates and non-negative amounts",
            ],
        }

    def resolve_client(self, identifier: str) -> dict[str, Any]:
        query = _normalise(identifier)
        if not query:
            raise ValueError("client identifier cannot be empty")
        exact = [
            client
            for client in self.clients
            if query in {_normalise(client["client_id"]), _normalise(client["name"])}
        ]
        if len(exact) == 1:
            return exact[0]
        partial = [
            client
            for client in self.clients
            if query in _normalise(client["client_id"]) or query in _normalise(client["name"])
        ]
        if len(partial) == 1:
            return partial[0]
        if not partial:
            raise LookupError(f"No client matches {identifier!r}")
        raise LookupError(
            f"Client identifier {identifier!r} is ambiguous: "
            + ", ".join(f"{client['client_id']} ({client['name']})" for client in partial)
        )

    def get_client_profile(self, identifier: str) -> StructuredResult:
        client = self.resolve_client(identifier)
        profile = {key: value for key, value in client.items() if key != "portfolio_holdings"}
        source = SourceReference(
            self._source_path(CLIENT_JSON), client["client_id"]
        )
        return StructuredResult(
            "client_profile",
            f"{client['name']} ({client['client_id']}) has risk profile "
            f"{client['risk_profile']} and risk score {client['risk_score_1_to_10']}.",
            profile,
            [source],
        )

    def get_holdings(self, identifier: str) -> StructuredResult:
        client = self.resolve_client(identifier)
        rows = self.holdings_by_client[client["client_id"]]
        data = [self._public_holding(row) for row in rows]
        sources = [
            SourceReference(
                self._source_path(HOLDINGS_CSV),
                f"{row['client_id']}:{row['product_name']}",
                row["_source_row"],
            )
            for row in rows
        ]
        return StructuredResult(
            "client_holdings",
            f"{client['name']} has {len(rows)} holdings totalling 100% of portfolio allocation.",
            data,
            sources,
        )

    @staticmethod
    def _public_holding(row: dict[str, Any]) -> dict[str, Any]:
        return {key: _clean_number(value) if isinstance(value, float) else value for key, value in row.items() if not key.startswith("_")}

    @staticmethod
    def _public_transaction(row: dict[str, Any]) -> dict[str, Any]:
        return {key: _clean_number(value) if isinstance(value, float) else value for key, value in row.items() if not key.startswith("_")}

    @staticmethod
    def _match_product(rows: Sequence[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        normalised = _normalise(query)
        exact = [row for row in rows if _normalise(row["product_name"]) == normalised]
        if exact:
            return exact
        partial = [row for row in rows if normalised in _normalise(row["product_name"])]
        products = sorted({row["product_name"] for row in partial})
        if not partial:
            raise LookupError(f"No product matches {query!r}")
        if len(products) > 1:
            raise LookupError(f"Product query {query!r} is ambiguous: {', '.join(products)}")
        return partial

    def calculate_product_exposure(
        self, identifier: str, product_query: str
    ) -> StructuredResult:
        client = self.resolve_client(identifier)
        matches = self._match_product(
            self.holdings_by_client[client["client_id"]], product_query
        )
        allocation = sum(row["allocation_pct"] for row in matches)
        value_sgd = sum(row["value_sgd"] for row in matches)
        product_name = matches[0]["product_name"]
        data = {
            "client_id": client["client_id"],
            "client_name": client["name"],
            "product_name": product_name,
            "allocation_pct": _clean_number(allocation),
            "value_sgd": _clean_number(value_sgd),
            "risk_profile": client["risk_profile"],
            "risk_score_1_to_10": client["risk_score_1_to_10"],
        }
        sources = [
            SourceReference(
                self._source_path(HOLDINGS_CSV),
                f"{row['client_id']}:{row['product_name']}",
                row["_source_row"],
            )
            for row in matches
        ]
        return StructuredResult(
            "product_exposure",
            f"{client['name']} has {allocation:g}% of the portfolio in {product_name}, "
            f"valued at SGD {value_sgd:,.0f}.",
            data,
            sources,
        )

    def calculate_asset_class_exposure(
        self, identifier: str, asset_class_query: str
    ) -> StructuredResult:
        client = self.resolve_client(identifier)
        query = _normalise(asset_class_query)
        matches = [
            row
            for row in self.holdings_by_client[client["client_id"]]
            if query in _normalise(row["asset_class"])
        ]
        if not matches:
            raise LookupError(f"No asset class matches {asset_class_query!r}")
        allocation = sum(row["allocation_pct"] for row in matches)
        value_sgd = sum(row["value_sgd"] for row in matches)
        data = {
            "client_id": client["client_id"],
            "client_name": client["name"],
            "asset_class_query": asset_class_query,
            "allocation_pct": _clean_number(allocation),
            "value_sgd": _clean_number(value_sgd),
            "matching_holdings": [self._public_holding(row) for row in matches],
        }
        sources = [
            SourceReference(
                self._source_path(HOLDINGS_CSV),
                f"{row['client_id']}:{row['product_name']}",
                row["_source_row"],
            )
            for row in matches
        ]
        return StructuredResult(
            "asset_class_exposure",
            f"{client['name']} has {allocation:g}% exposure matching asset class "
            f"{asset_class_query!r}, valued at SGD {value_sgd:,.0f}.",
            data,
            sources,
        )

    def get_transactions(
        self,
        identifier: str,
        status: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> StructuredResult:
        client = self.resolve_client(identifier)
        start = date.fromisoformat(start_date) if start_date else None
        end = date.fromisoformat(end_date) if end_date else None
        if start and end and start > end:
            raise ValueError("start_date cannot be after end_date")
        rows = []
        for row in self.transactions_by_client[client["client_id"]]:
            row_date = date.fromisoformat(row["date"])
            if status and _normalise(row["status"]) != _normalise(status):
                continue
            if start and row_date < start:
                continue
            if end and row_date > end:
                continue
            rows.append(row)
        rows.sort(key=lambda row: (row["date"], row["transaction_id"]))
        data = [self._public_transaction(row) for row in rows]
        sources = [
            SourceReference(
                self._source_path(TRANSACTIONS_CSV), row["transaction_id"], row["_source_row"]
            )
            for row in rows
        ]
        qualifier = f" with status {status}" if status else ""
        return StructuredResult(
            "client_transactions",
            f"Found {len(rows)} transactions for {client['name']}{qualifier}.",
            data,
            sources,
        )

    def find_clients_by_risk_profile(self, risk_profile: str) -> StructuredResult:
        query = _normalise(risk_profile)
        matches = [
            client for client in self.clients if query in _normalise(client["risk_profile"])
        ]
        data = [
            {
                "client_id": client["client_id"],
                "name": client["name"],
                "risk_profile": client["risk_profile"],
                "risk_score_1_to_10": client["risk_score_1_to_10"],
                "investor_status": client["investor_status"],
            }
            for client in matches
        ]
        sources = [
            SourceReference(self._source_path(CLIENT_JSON), client["client_id"])
            for client in matches
        ]
        return StructuredResult(
            "clients_by_risk_profile",
            f"Found {len(matches)} clients whose risk profile matches {risk_profile!r}.",
            data,
            sources,
        )

    @staticmethod
    def is_complex_holding(holding: dict[str, Any]) -> bool:
        classification = _normalise(holding["asset_class"])
        return any(keyword in classification for keyword in COMPLEX_ASSET_KEYWORDS)

    def find_complex_product_concentrations(
        self, threshold_pct: float = 20.0, retail_only: bool = False
    ) -> StructuredResult:
        if threshold_pct < 0:
            raise ValueError("threshold_pct cannot be negative")
        matches: list[dict[str, Any]] = []
        sources: list[SourceReference] = []
        for holding in self.holdings:
            client = self.clients_by_id[holding["client_id"]]
            if retail_only and "retail investor" not in _normalise(client["investor_status"]):
                continue
            if self.is_complex_holding(holding) and holding["allocation_pct"] > threshold_pct:
                matches.append(
                    {
                        "client_id": client["client_id"],
                        "client_name": client["name"],
                        "investor_status": client["investor_status"],
                        "risk_profile": client["risk_profile"],
                        "product_name": holding["product_name"],
                        "asset_class": holding["asset_class"],
                        "allocation_pct": _clean_number(holding["allocation_pct"]),
                        "value_sgd": _clean_number(holding["value_sgd"]),
                    }
                )
                sources.append(
                    SourceReference(
                        self._source_path(HOLDINGS_CSV),
                        f"{holding['client_id']}:{holding['product_name']}",
                        holding["_source_row"],
                    )
                )
        matches.sort(key=lambda row: (-row["allocation_pct"], row["client_id"]))
        scope = "Retail Investors" if retail_only else "all investors"
        return StructuredResult(
            "complex_product_concentrations",
            f"Found {len(matches)} Complex Product holdings above {threshold_pct:g}% among {scope}. "
            "Classification uses explicit Complex/Structured/SIP wording in the holdings asset class.",
            matches,
            sources,
        )

    def validation_result(self) -> StructuredResult:
        return StructuredResult(
            "validate",
            f"Validated {self.validation['client_count']} clients, "
            f"{self.validation['holding_count']} holdings, and "
            f"{self.validation['transaction_count']} transactions.",
            self.validation,
            [
                SourceReference(self._source_path(CLIENT_JSON), "clients"),
                SourceReference(self._source_path(HOLDINGS_CSV), "holdings"),
                SourceReference(self._source_path(TRANSACTIONS_CSV), "transactions"),
            ],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query authoritative structured wealth data")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")

    client = commands.add_parser("client")
    client.add_argument("identifier")
    holdings = commands.add_parser("holdings")
    holdings.add_argument("identifier")
    exposure = commands.add_parser("exposure")
    exposure.add_argument("identifier")
    exposure.add_argument("product")
    asset = commands.add_parser("asset-exposure")
    asset.add_argument("identifier")
    asset.add_argument("asset_class")
    transactions = commands.add_parser("transactions")
    transactions.add_argument("identifier")
    transactions.add_argument("--status")
    transactions.add_argument("--start-date")
    transactions.add_argument("--end-date")
    risk = commands.add_parser("risk-profile")
    risk.add_argument("profile")
    concentrations = commands.add_parser("concentrations")
    concentrations.add_argument("--threshold", type=float, default=20.0)
    concentrations.add_argument("--retail-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = StructuredDataStore(args.data_dir)
        if args.command == "validate":
            result = store.validation_result()
        elif args.command == "client":
            result = store.get_client_profile(args.identifier)
        elif args.command == "holdings":
            result = store.get_holdings(args.identifier)
        elif args.command == "exposure":
            result = store.calculate_product_exposure(args.identifier, args.product)
        elif args.command == "asset-exposure":
            result = store.calculate_asset_class_exposure(args.identifier, args.asset_class)
        elif args.command == "transactions":
            result = store.get_transactions(
                args.identifier, args.status, args.start_date, args.end_date
            )
        elif args.command == "risk-profile":
            result = store.find_clients_by_risk_profile(args.profile)
        else:
            result = store.find_complex_product_concentrations(
                args.threshold, args.retail_only
            )
    except Exception as exc:
        print(f"Structured data query failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
