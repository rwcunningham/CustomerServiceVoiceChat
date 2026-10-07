import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from .config import Settings


SECTION_RE = re.compile(
    r"(?ms)^=+\s*\n"
    r"(?P<number>\d{2})\.\s+(?P<title>[^\n]+)\n"
    r"=+\s*\n"
    r"URL:\s*(?P<url>[^\n]+)\n"
    r"(?P<body>.*?)(?=^=+\s*\n(?:\d{2}\.|SHARED SITE ELEMENTS|RELATED DESTINATIONS)|\Z)"
)


def chunk_words(text: str, chunk_words: int = 280, overlap_words: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks: list[str] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def parse_page_sections(text: str) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for match in SECTION_RE.finditer(text):
        number = match.group("number").strip()
        title = match.group("title").strip()
        url = match.group("url").strip()
        body = match.group("body").strip()
        section_text = f"Page {number}: {title}\nURL: {url}\n{body}".strip()
        sections.append({
            "section_id": number,
            "title": title,
            "url": url,
            "text": section_text,
        })
    return sections


class KnowledgeBase:
    """Small, inspectable semantic knowledge base for the restaurant assistant."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.db_path = Path(settings.db_path)

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
            if "metadata" not in columns:
                conn.execute(
                    "ALTER TABLE chunks ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'"
                )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
            conn.commit()

    async def _embed_many(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start+64]
            response = await self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
                dimensions=self.settings.embedding_dimensions,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return vectors

    @staticmethod
    def _normalize(vector: list[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        return arr if norm == 0.0 else arr / norm

    async def _write_chunks(self, items: list[dict[str, Any]], source: str, replace_source: bool) -> dict[str, Any]:
        if not items:
            return {"source": source, "chunks_written": 0}

        raw_vectors = await self._embed_many([item["content"] for item in items])
        vectors = [self._normalize(v).tolist() for v in raw_vectors]
        rows = []
        for item, vector in zip(items, vectors):
            content = item["content"]
            digest = hashlib.sha256(f"{source}\0{content}".encode("utf-8")).hexdigest()
            rows.append((
                source,
                content,
                digest,
                json.dumps(vector, separators=(",", ":")),
                json.dumps(item.get("metadata", {}), ensure_ascii=False, separators=(",", ":")),
            ))

        with sqlite3.connect(self.db_path) as conn:
            if replace_source:
                conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            conn.executemany(
                """INSERT OR IGNORE INTO chunks
                   (source, content, content_hash, embedding, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
        return {"source": source, "chunks_written": len(rows)}

    async def ingest_text(self, text: str, source: str, replace_source: bool = True) -> dict[str, Any]:
        items = [{"content": c, "metadata": {"kind": "generic_text"}} for c in chunk_words(text)]
        return await self._write_chunks(items, source, replace_source)

    async def ingest_page_summary_document(self, text: str, source: str, replace_source: bool = True) -> dict[str, Any]:
        sections = parse_page_sections(text)
        if not sections:
            result = await self.ingest_text(text, source, replace_source)
            result["mode"] = "generic_fallback"
            return result

        items: list[dict[str, Any]] = []
        for section in sections:
            chunks = chunk_words(section["text"], chunk_words=260, overlap_words=40)
            for index, chunk in enumerate(chunks, start=1):
                items.append({
                    "content": chunk,
                    "metadata": {
                        "kind": "page_summary",
                        "section_id": section["section_id"],
                        "title": section["title"],
                        "url": section["url"],
                        "chunk_index": index,
                    },
                })

        result = await self._write_chunks(items, source, replace_source)
        result.update({"mode": "page_aware", "sections_found": len(sections)})
        return result

    async def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        top_k = max(1, min(int(top_k), 8))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, source, content, embedding, metadata FROM chunks"
            ).fetchall()

        if not rows:
            return {"query": query, "matches": [], "note": "The knowledge base is empty."}

        response = await self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=query,
            dimensions=self.settings.embedding_dimensions,
            encoding_format="float",
        )
        q = self._normalize(response.data[0].embedding)
        embeddings = np.asarray([json.loads(row[3]) for row in rows], dtype=np.float32)
        scores = embeddings @ q
        indices = np.argsort(scores)[::-1][:top_k]

        matches = []
        for idx in indices:
            row = rows[int(idx)]
            try:
                metadata = json.loads(row[4] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            matches.append({
                "chunk_id": row[0],
                "source": row[1],
                "score": round(float(scores[int(idx)]), 4),
                "metadata": metadata,
                "text": row[2],
            })

        return {
            "query": query,
            "matches": matches,
            "note": (
                "Semantic retrieval only. Use a passage only when it supports the requested fact. "
                "Historical targets or undated figures are not proof of current status, and missing "
                "product-level details must not be inferred."
            ),
        }

    def stats(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            sources = [
                {"source": row[0], "chunks": row[1]}
                for row in conn.execute(
                    "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
                ).fetchall()
            ]
        return {"total_chunks": total, "sources": sources}
