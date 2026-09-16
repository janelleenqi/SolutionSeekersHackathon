"""Citation-ready document ingestion for the Wealth Advisor Assistant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "1.0"
DEFAULT_CHUNK_SIZE = 1_200
DEFAULT_CHUNK_OVERLAP = 180
SUPPORTED_SUFFIXES = {".pdf", ".json", ".csv", ".txt", ".md"}
STRUCTURED_FILENAMES = {"clients_portfolio.json", "clients_portfolio.csv", "transactions.csv"}


@dataclass(frozen=True)
class IngestionConfig:
    input_dir: Path
    output_dir: Path
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    include_structured: bool = False

    def validate(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": self.metadata}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(text: str) -> str:
    """Normalize extraction noise while preserving evidence-bearing wording."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    text = re.sub(r"[\t\u00a0 ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _text_units(text: str, max_size: int) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", clean_text(text)):
        if not paragraph:
            continue
        if len(paragraph) <= max_size:
            units.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
        for sentence in sentences:
            if len(sentence) <= max_size:
                if sentence:
                    units.append(sentence)
                continue
            start = 0
            while start < len(sentence):
                end = min(start + max_size, len(sentence))
                if end < len(sentence):
                    boundary = sentence.rfind(" ", start, end)
                    if boundary > start + max_size // 2:
                        end = boundary
                units.append(sentence[start:end].strip())
                start = end
    return units


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Make paragraph-aware chunks with a bounded character overlap."""
    units = _text_units(text, chunk_size)
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        proposed = "\n\n".join([*current, unit])
        if current and len(proposed) > chunk_size:
            chunks.append("\n\n".join(current))
            carry: list[str] = []
            for previous in reversed(current):
                candidate = "\n\n".join([previous, *carry])
                if carry and len(candidate) > overlap:
                    break
                if not carry and len(previous) > overlap:
                    previous = previous[-overlap:].lstrip()
                carry.insert(0, previous)
                if len("\n\n".join(carry)) >= overlap:
                    break
            current = carry
        current.append(unit)
    if current:
        final = "\n\n".join(current)
        if not chunks or final != chunks[-1]:
            chunks.append(final)
    return chunks


def classify_document(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "fund_factsheets" in parts:
        return "fund_factsheet"
    if "policy" in parts:
        return "policy"
    if name == "client_correspondence.json":
        return "client_correspondence"
    if "complaint" in name:
        return "client_complaint"
    if "call_notes" in name:
        return "rm_call_note"
    if "acknowledgement" in name:
        return "risk_acknowledgement"
    if name.startswith("clients_portfolio"):
        return "client_portfolio"
    if name == "transactions.csv":
        return "transaction"
    return "document"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _make_chunks(
    text: str,
    metadata: dict[str, Any],
    source_hash: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    result: list[Chunk] = []
    metadata = {key: value for key, value in metadata.items() if value not in (None, "")}
    for index, value in enumerate(chunk_text(text, chunk_size, overlap)):
        content_hash = sha256_bytes(value.encode("utf-8"))
        locator = "|".join(
            str(metadata.get(key, "")) for key in ("source_path", "page", "record_id")
        )
        chunk_id = hashlib.sha256(f"{locator}|{index}|{content_hash}".encode()).hexdigest()[:24]
        result.append(
            Chunk(
                id=chunk_id,
                text=value,
                metadata={
                    **metadata,
                    "chunk_index": index,
                    "source_sha256": source_hash,
                    "content_sha256": content_hash,
                    "schema_version": SCHEMA_VERSION,
                },
            )
        )
    return result


def _extract_pdf_pages(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf; run pip install -r requirements.txt") from exc
    pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
    repeated: set[str] = set()
    if len(pages) >= 3:
        candidates: Counter[str] = Counter()
        for page in pages:
            lines = [line.strip() for line in clean_text(page).splitlines() if line.strip()]
            candidates.update(set(lines[:2] + lines[-2:]))
        threshold = max(3, (len(pages) + 1) // 2)
        repeated = {
            line for line, count in candidates.items() if count >= threshold and len(line) < 120
        }
    return [
        clean_text("\n".join(line for line in clean_text(page).splitlines() if line.strip() not in repeated))
        for page in pages
    ]


def ingest_pdf(path: Path, config: IngestionConfig, source_hash: str) -> list[Chunk]:
    result: list[Chunk] = []
    for page_number, text in enumerate(_extract_pdf_pages(path), start=1):
        if text:
            result.extend(
                _make_chunks(
                    text,
                    {
                        "source_path": _relative(path, config.input_dir),
                        "source_filename": path.name,
                        "file_type": "pdf",
                        "document_type": classify_document(path),
                        "page": page_number,
                        "title": path.stem.replace("_", " ").title(),
                    },
                    source_hash,
                    config.chunk_size,
                    config.chunk_overlap,
                )
            )
    return result


def _value_to_text(value: Any, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            label = key.replace("_", " ").title()
            if isinstance(item, (dict, list)):
                lines.extend((f"{prefix}{label}:", _value_to_text(item, indent + 1)))
            else:
                lines.append(f"{prefix}{label}: {item}")
        return "\n".join(lines)
    if isinstance(value, list):
        return "\n".join(f"{prefix}- {_value_to_text(item, indent + 1).lstrip()}" for item in value)
    return f"{prefix}{value}"


def ingest_json(path: Path, config: IngestionConfig, source_hash: str) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    common = {
        "source_path": _relative(path, config.input_dir),
        "source_filename": path.name,
        "file_type": "json",
    }
    result: list[Chunk] = []
    if path.name.lower() == "client_correspondence.json":
        for thread in data.get("email_threads", []):
            thread_id = str(thread.get("thread_id", "unknown"))
            subject = str(thread.get("subject", "(no subject)"))
            for index, message in enumerate(thread.get("messages", []), start=1):
                text = (
                    f"Email thread: {thread_id}\nSubject: {subject}\n"
                    f"From: {message.get('from', '')}\nTo: {message.get('to', '')}\n"
                    f"Date: {message.get('date', '')}\n\n{message.get('body', '')}"
                )
                result.extend(
                    _make_chunks(
                        text,
                        {
                            **common,
                            "document_type": "client_correspondence",
                            "record_id": f"{thread_id}-M{index}",
                            "thread_id": thread_id,
                            "message_index": index,
                            "client_id": thread.get("related_client_id"),
                            "subject": subject,
                            "date": message.get("date"),
                            "sender": message.get("from"),
                            "recipient": message.get("to"),
                        },
                        source_hash,
                        config.chunk_size,
                        config.chunk_overlap,
                    )
                )
        return result
    records = data.get("clients", []) if path.name.lower() == "clients_portfolio.json" else (
        data if isinstance(data, list) else [data]
    )
    for index, record in enumerate(records, start=1):
        record_id = str(record.get("client_id", index)) if isinstance(record, dict) else str(index)
        result.extend(
            _make_chunks(
                _value_to_text(record),
                {
                    **common,
                    "document_type": classify_document(path),
                    "record_id": record_id,
                    "client_id": record.get("client_id") if isinstance(record, dict) else None,
                    "client_name": record.get("name") if isinstance(record, dict) else None,
                    "risk_profile": record.get("risk_profile") if isinstance(record, dict) else None,
                },
                source_hash,
                config.chunk_size,
                config.chunk_overlap,
            )
        )
    return result


def ingest_csv(path: Path, config: IngestionConfig, source_hash: str) -> list[Chunk]:
    result: list[Chunk] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            result.extend(
                _make_chunks(
                    _value_to_text(row),
                    {
                        "source_path": _relative(path, config.input_dir),
                        "source_filename": path.name,
                        "file_type": "csv",
                        "document_type": classify_document(path),
                        "record_id": row.get("transaction_id") or f"row-{row_number}",
                        "row_number": row_number,
                        "client_id": row.get("client_id"),
                        "client_name": row.get("name"),
                        "product_name": row.get("product_name"),
                        "date": row.get("date"),
                        "status": row.get("status"),
                    },
                    source_hash,
                    config.chunk_size,
                    config.chunk_overlap,
                )
            )
    return result


def ingest_file(path: Path, config: IngestionConfig) -> list[Chunk]:
    source_hash = sha256_bytes(path.read_bytes())
    if path.suffix.lower() == ".pdf":
        return ingest_pdf(path, config, source_hash)
    if path.suffix.lower() == ".json":
        return ingest_json(path, config, source_hash)
    if path.suffix.lower() == ".csv":
        return ingest_csv(path, config, source_hash)
    return _make_chunks(
        path.read_text(encoding="utf-8-sig"),
        {
            "source_path": _relative(path, config.input_dir),
            "source_filename": path.name,
            "file_type": path.suffix.lower().lstrip("."),
            "document_type": classify_document(path),
        },
        source_hash,
        config.chunk_size,
        config.chunk_overlap,
    )


def discover_files(config: IngestionConfig) -> list[Path]:
    files = [
        path for path in config.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not config.include_structured:
        files = [path for path in files if path.name.lower() not in STRUCTURED_FILENAMES]
    return sorted(files, key=lambda path: _relative(path, config.input_dir).lower())


def run_ingestion(config: IngestionConfig) -> dict[str, Any]:
    config.validate()
    if not config.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {config.input_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    errors: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    files = discover_files(config)
    for path in files:
        source = _relative(path, config.input_dir)
        try:
            new_chunks = ingest_file(path, config)
            chunks.extend(new_chunks)
            source_counts[source] = len(new_chunks)
        except Exception as exc:
            errors.append({"source_path": source, "error": f"{type(exc).__name__}: {exc}"})
    chunks_path = config.output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    type_counts = Counter(chunk.metadata["document_type"] for chunk in chunks)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(config.input_dir.resolve()),
        "output_file": chunks_path.name,
        "settings": {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "include_structured": config.include_structured,
        },
        "files_discovered": len(files),
        "files_succeeded": len(source_counts),
        "total_chunks": len(chunks),
        "chunks_by_document_type": dict(sorted(type_counts.items())),
        "chunks_by_source": source_counts,
        "errors": errors,
    }
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build citation-ready chunks from raw wealth data")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--include-structured", action="store_true")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_ingestion(
            IngestionConfig(
                args.input, args.output, args.chunk_size, args.chunk_overlap, args.include_structured
            )
        )
        if args.database:
            from src.database import load_database

            manifest["database"] = load_database(args.input, args.database)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 1 if args.strict and manifest["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
