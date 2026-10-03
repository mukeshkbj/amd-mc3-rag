"""Prompt -> model JSON -> validated answer + exact-set citations.

Cite only what is necessary: the model names the files it used, we normalize
them to corpus-relative paths and intersect with what was actually retrieved.
Refusal is an empty answer AND an empty citation list — never prose.
"""
from __future__ import annotations

import json
import re

from indexer import Chunk
from retriever import BM25, retrieve

_PROMPT = """You are a precise document QA system. Answer ONLY from the context below.
The products described are fictional; anything you know from elsewhere is wrong.

Context:
{context}

Question: {question}

Reply with EXACTLY one JSON object, nothing else:
{{"answer": "<the value only>", "files": ["<path>", ...]}}
or {{"answer": "", "files": []}} if the context does not contain the answer.

Rules:
- The value only: a number, part number, version, quarter — with its full
  qualifier (Q3 FY27 not Q3, REV-C2 not C2). No sentences, no units explanation.
- Cite a file ONLY if removing it would make the answer impossible. A file that
  merely discusses the topic is not a source.
- If the question describes content in one file that links to the answer in
  another (e.g. a log naming a ticket id that a table maps to a fix), cite BOTH.
- In source code prefer constants over comments.
- Never answer from memory; if no context file contains it, refuse."""

_JSON = re.compile(r"\{.*\}", re.S)
_NORM = re.compile(r"[\s\-\.·_]")


def normalize(s: str) -> str:
    return _NORM.sub("", s.upper())


def _extract(raw: str) -> dict | None:
    m = _JSON.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def answer(engine, bm25: BM25, query: str) -> tuple[str, list[str], float]:
    hits, bridge = retrieve(bm25, query)
    if not hits:
        return "", [], 0.0
    known = {c.file for c in bm25.chunks}
    ctx_parts, cited_files = [], set()
    for _s, i in hits:
        c = bm25.chunks[i]
        cited_files.add(c.file)
        ctx_parts.append(f"[F:{c.file}]\n{c.text}")
    raw = engine.generate(
        _PROMPT.format(context="\n\n".join(ctx_parts), question=query),
        max_new_tokens=256)
    obj = _extract(raw)
    if not obj:
        return "", [], 0.0
    ans = str(obj.get("answer", "")).strip().strip('"')
    files = obj.get("files", [])
    if not isinstance(files, list):
        files = [files]
    cites = []
    for f in files:
        rel = str(f).replace("\\", "/")
        rel = rel.split("/app/corpus/", 1)[-1].lstrip("/")
        if rel in known and rel in cited_files and rel not in cites:
            cites.append(rel)
    if not ans:
        return "", [], 0.0
    if not cites:
        # model answered but named nothing usable: cite only the best-scoring
        # file whose retrieved text actually contains the answer token
        want = normalize(ans)
        for _s, i in hits:
            c = bm25.chunks[i]
            if want and want in normalize(c.text):
                cites = [c.file]
                break
    for c in list(cites):
        for g in bridge.get(c, ()):
            if g in cited_files and g not in cites:
                cites.append(g)
    return ans, cites, 0.9 if cites else 0.4
