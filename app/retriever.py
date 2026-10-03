"""BM25 retrieval + identifier-expansion second pass (the multi-hop route:
log names the ticket, a different file carries its resolution).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from indexer import Chunk

TOKEN = re.compile(r"[A-Za-z0-9]+|[一-鿿]")
IDENT = re.compile(
    r"\b[A-Z]{1,4}-[A-Z0-9]+(?:-[A-Z0-9]+)*\b"   # TQ-40, ORR-1847, ORR-FAN-2214-B, REV-C2
    r"|\b[A-Z]\d{3,}\b"                          # E7731
    r"|\b\d+\.\d+(?:\.\d+)+\b"                   # 4.3.2
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.dl = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.dl) / max(1, len(chunks))
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5))
                    for w, f in df.items()}

    def top(self, query: str, k: int = 10, include_dead: bool = False):
        q = Counter(tokenize(query))
        scored = []
        for i, t in enumerate(self.tf):
            if self.chunks[i].dead and not include_dead:
                continue
            s = 0.0
            for w in q:
                c = t.get(w, 0)
                if not c:
                    continue
                s += self.idf.get(w, 0.0) * c * (self.k1 + 1) / (
                    c + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return scored[:k]


def identifiers(text: str) -> set[str]:
    return set(IDENT.findall(text))


def retrieve(bm25: BM25, query: str, k: int = 10, max_files: int = 5):
    """Two passes: question -> chunks; identifiers found in them -> more chunks.
    Dead (superseded) chunks are excluded from context entirely.

    Returns (hits, bridge). bridge[f] = files whose retrieved chunks supplied
    an identifier that reached f in the second pass — i.e. f is only
    answerable *through* those files, so they are necessary citations too
    (the log names the ticket; a different file carries its resolution)."""
    first = bm25.top(query, k=k)
    ids = set()
    for _s, i in first[:6]:
        ids |= identifiers(bm25.chunks[i].text)
    hits = list(first)
    if ids:
        second = bm25.top(" ".join(sorted(ids)), k=k)
        seen = {i for _s, i in hits}
        hits += [(s, i) for s, i in second if i not in seen]
    hits.sort(reverse=True)
    files: list[str] = []
    picked = []
    for s, i in hits:
        f = bm25.chunks[i].file
        if f not in files:
            if len(files) >= max_files:
                continue
            files.append(f)
        picked.append((s, i))

    # bridge map: a cited file f may only be answerable via a rare join key
    # (ticket id, error code, part no.) that some other file g supplied.
    # g is a necessary citation only if g is a top direct query match — the
    # file the question is actually about. Version strings (x.y.z) are values
    # found via the join, never keys, so they are excluded.
    q_ids = identifiers(query)
    entry_files = set()
    for _s, i in first[:3]:
        entry_files.add(bm25.chunks[i].file)
    ident_files: dict[str, set[str]] = {}
    file_ids: dict[str, set[str]] = {}
    for c in bm25.chunks:
        for ident in identifiers(c.text):
            if re.fullmatch(r"\d+(?:\.\d+)+", ident):
                continue
            ident_files.setdefault(ident, set()).add(c.file)
            file_ids.setdefault(c.file, set()).add(ident)
    rare = {i for i, fs in ident_files.items() if len(fs) <= 2} - q_ids
    bridge: dict[str, set[str]] = {}
    for f in files:
        linked = {g for g in entry_files if g != f
                  and file_ids.get(g, set()) & file_ids.get(f, set()) & rare}
        if linked:
            bridge[f] = linked
    return picked[:14], bridge
