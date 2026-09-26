"""Document ingestion: PDF -> cleaned page text -> overlapping word chunks -> embeddings -> Chroma.

Run:  uv run python -m assistant.rag.ingest            (idempotent; rebuilds the collection)
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass

import yaml
from pypdf import PdfReader

from ..config import Settings, get_settings

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    paper_id: str
    title: str
    page: int
    text: str


def load_catalog(s: Settings) -> dict[str, dict]:
    return yaml.safe_load(s.catalog_path.read_text(encoding="utf-8"))


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)      # join lines
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


REFERENCES_RE = re.compile(r"\n\s*(REFERENCES|References|Bibliography|BIBLIOGRAPHY)\s*\n")


def chunk_pages(paper_id: str, title: str, pages: list[str], words: int, overlap: int) -> list[Chunk]:
    """Sliding word window *within* a page so every chunk has an exact page citation.
    Pages that start the bibliography (and all after) are dropped: they add noise to retrieval."""
    out: list[Chunk] = []
    in_refs = False
    for pno, raw in enumerate(pages, start=1):
        if in_refs:
            break
        m = REFERENCES_RE.search(raw)
        if m and pno > 2:
            raw, in_refs = raw[: m.start()], True
        toks = clean(raw).split()
        if len(toks) < 30:
            continue
        step = max(1, words - overlap)
        for n, start in enumerate(range(0, len(toks), step)):
            piece = toks[start:start + words]
            if len(piece) < 40 and n > 0:
                break
            out.append(Chunk(f"{paper_id}:p{pno}:c{n}", paper_id, title, pno, " ".join(piece)))
    return out


def build_chunks(s: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pid, meta in load_catalog(s).items():
        path = s.papers_dir / meta["file"]
        if not path.exists():
            log.warning("missing %s -- skipped", path)
            continue
        pages = [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
        cs = chunk_pages(pid, meta["title"], pages, s.chunk_words, s.chunk_overlap_words)
        log.info("%-16s %3d pages -> %4d chunks", pid, len(pages), len(cs))
        chunks += cs
    return chunks


def ingest(s: Settings | None = None) -> int:
    from .store import VectorStore

    s = s or get_settings()
    t0 = time.time()
    chunks = build_chunks(s)
    store = VectorStore(s)
    store.rebuild(chunks)
    log.info("ingested %d chunks in %.1fs", len(chunks), time.time() - t0)
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-words", type=int)
    a = ap.parse_args()
    st = get_settings()
    if a.chunk_words:
        st.chunk_words = a.chunk_words
    ingest(st)
