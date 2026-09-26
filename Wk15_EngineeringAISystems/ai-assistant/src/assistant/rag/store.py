"""Vector store (Chroma, persistent) + hybrid retrieval (dense bge-small + BM25, fused with RRF)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import chromadb
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from ..config import Settings, get_settings

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(t: str) -> list[str]:
    return _TOKEN.findall(t.lower())


@dataclass
class Hit:
    chunk_id: str
    paper_id: str
    title: str
    page: int
    text: str
    score: float

    def to_dict(self, char_cap: int | None = None) -> dict:
        txt = self.text if not char_cap or len(self.text) <= char_cap else self.text[:char_cap] + " …"
        return {"chunk_id": self.chunk_id, "paper_id": self.paper_id, "page": self.page,
                "score": round(self.score, 4), "text": txt}


class VectorStore:
    def __init__(self, s: Settings | None = None):
        self.s = s or get_settings()
        self.s.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.s.chroma_dir))
        self.embedder = TextEmbedding(self.s.embed_model)
        self.col = self.client.get_or_create_collection(self.s.collection, metadata={"hnsw:space": "cosine"})
        self._bm25 = None
        self._docs: list[dict] = []

    # ---------------------------------------------------------- write
    def rebuild(self, chunks) -> None:
        try:
            self.client.delete_collection(self.s.collection)
        except Exception:  # noqa: BLE001
            pass
        self.col = self.client.get_or_create_collection(self.s.collection, metadata={"hnsw:space": "cosine"})
        B = 128
        for i in range(0, len(chunks), B):
            batch = chunks[i:i + B]
            texts = [c.text for c in batch]
            embs = [e.tolist() for e in self.embedder.passage_embed(texts)]
            self.col.add(
                ids=[c.id for c in batch], documents=texts, embeddings=embs,
                metadatas=[{"paper_id": c.paper_id, "title": c.title, "page": c.page} for c in batch],
            )
        self._bm25 = None

    # ---------------------------------------------------------- read
    def count(self) -> int:
        return self.col.count()

    def _ensure_bm25(self):
        if self._bm25 is None:
            got = self.col.get(include=["documents", "metadatas"])
            self._docs = [{"id": i, "text": d, **m} for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]
            self._bm25 = BM25Okapi([_tok(d["text"]) for d in self._docs]) if self._docs else None

    def get(self, chunk_id: str) -> Hit | None:
        got = self.col.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not got["ids"]:
            return None
        m = got["metadatas"][0]
        return Hit(chunk_id, m["paper_id"], m["title"], int(m["page"]), got["documents"][0], 1.0)

    def search(self, query: str, k: int | None = None, paper_id: str | None = None) -> list[Hit]:
        """Hybrid search: dense top-N and BM25 top-N fused with reciprocal-rank fusion, capped at max_top_k."""
        k = min(k or self.s.top_k, self.s.max_top_k)
        n = max(k * 4, 20)
        where = {"paper_id": paper_id} if paper_id else None
        qemb = next(iter(self.embedder.query_embed(query))).tolist()
        dense = self.col.query(query_embeddings=[qemb], n_results=n, where=where,
                               include=["documents", "metadatas", "distances"])
        cand: dict[str, dict] = {}
        for rank, (cid, doc, meta) in enumerate(zip(dense["ids"][0], dense["documents"][0], dense["metadatas"][0])):
            cand[cid] = {"id": cid, "text": doc, **meta, "rrf": 1 / (60 + rank)}
        self._ensure_bm25()
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tok(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            r = 0
            for i in order:
                d = self._docs[i]
                if paper_id and d["paper_id"] != paper_id:
                    continue
                cand.setdefault(d["id"], {**d, "rrf": 0.0})["rrf"] += 1 / (60 + r)
                r += 1
                if r >= n:
                    break
        best = sorted(cand.values(), key=lambda c: -c["rrf"])[:k]
        return [Hit(c["id"], c["paper_id"], c["title"], int(c["page"]), c["text"], c["rrf"]) for c in best]
