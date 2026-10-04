"""Document loaders for plain-text, markdown and PDF files."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".pdf"}


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def load_document(path: Path) -> str:
    """Return the raw text content of a supported file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    return path.read_text(encoding="utf-8", errors="replace")