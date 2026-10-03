"""Corpus walk -> chunks -> persisted index. One bad file never aborts the walk:
every parse is wrapped, failures are recorded, and the walk carries on.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from parsers import Unreadable, parse_file

INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
CHUNK_CHARS = 900


@dataclass
class Chunk:
    file: str      # corpus-relative, forward slashes
    idx: int       # chunk number within file
    text: str
    dead: bool     # withdrawn/superseded source


def _pack(segments: list[str], file: str, dead: bool) -> list[Chunk]:
    chunks, buf, idx = [], [], 0
    size = 0
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if size + len(seg) > CHUNK_CHARS and buf:
            chunks.append(Chunk(file, idx, "\n".join(buf), dead)); idx += 1
            buf, size = [], 0
        buf.append(seg); size += len(seg) + 1
    if buf:
        chunks.append(Chunk(file, idx, "\n".join(buf), dead))
    return chunks


def build(corpus: Path, engine=None) -> tuple[list[Chunk], list[dict]]:
    corpus = Path(corpus)
    chunks, skipped = [], []
    for dirpath, _dirs, files in os.walk(corpus):
        for name in sorted(files):
            p = Path(dirpath) / name
            rel = p.relative_to(corpus).as_posix()
            try:
                segs, dead = parse_file(p, engine=engine)
            except Exception as e:  # Unreadable, PermissionError, anything
                skipped.append({"file": rel, "reason": str(e)[:200]})
                continue
            chunks.extend(_pack(segs, rel, dead))
    return chunks, skipped


def save(chunks: list[Chunk], skipped: list[dict], index_dir: Path = INDEX_DIR):
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in chunks),
        encoding="utf-8")
    (index_dir / "skipped.json").write_text(
        json.dumps(skipped, indent=1), encoding="utf-8")


def load(index_dir: Path = INDEX_DIR) -> list[Chunk] | None:
    f = index_dir / "chunks.jsonl"
    if not f.exists():
        return None
    return [Chunk(**json.loads(l))
            for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
