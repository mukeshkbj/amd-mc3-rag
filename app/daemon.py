"""Resident MC3 server: loads the VLM once (startup budget), holds the corpus
index in memory, serves --index and --query RPCs over a socket.

app.py is a thin client per harness exec; the model and index never reload.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import traceback
from pathlib import Path

READY_PATH = Path(os.environ.get("MC3_READY", "/tmp/mc3-rag.ready"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))


def _addr():
    raw = os.environ.get("MC3_SOCK", "/tmp/mc3-rag.sock")
    if raw.startswith("tcp:"):
        host, port = raw[4:].rsplit(":", 1)
        return socket.AF_INET, (host, int(port)), None
    return socket.AF_UNIX, raw, raw


class Rag:
    def __init__(self) -> None:
        from engine import VLMEngine
        self.engine = VLMEngine()
        self.engine.warmup()
        self.chunks = None
        self.bm25 = None
        self.corpus = None

    def index(self, corpus: str) -> dict:
        import indexer
        from retriever import BM25

        self.corpus = corpus
        self.chunks, skipped = indexer.build(Path(corpus), engine=self.engine)
        indexer.save(self.chunks, skipped, INDEX_DIR)
        self.bm25 = BM25(self.chunks)
        return {"ok": True, "chunks": len(self.chunks), "skipped": len(skipped)}

    def _ensure_indexed(self, corpus: str) -> None:
        import indexer
        from retriever import BM25

        if self.bm25 is not None and self.corpus == corpus:
            return
        cached = indexer.load(INDEX_DIR)
        if cached:
            self.chunks, self.corpus = cached, corpus
            self.bm25 = BM25(self.chunks)
            return
        self.index(corpus)

    def query(self, corpus: str, question: str) -> dict:
        from answerer import answer

        self._ensure_indexed(corpus)
        text, cites, conf = answer(self.engine, self.bm25, question)
        return {"answer": text, "citations": cites, "confidence": conf}


def _handle(rag: Rag, req: dict) -> dict:
    if req.get("cmd") == "index":
        return rag.index(req["corpus"])
    if req.get("cmd") == "query":
        return rag.query(req["corpus"], req["query"])
    return {"error": "unknown cmd"}


def main() -> int:
    # marker exists from process start (before model load) so app.py can tell
    # "daemon loading" apart from "no daemon" and doesn't spawn a duplicate
    Path(str(READY_PATH) + ".loading").write_text(str(os.getpid()))
    rag = Rag()
    family, addr, sock_file = _addr()
    if sock_file and Path(sock_file).exists():
        Path(sock_file).unlink()
    srv = socket.socket(family, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(addr)
    srv.listen(4)
    if sock_file:
        os.chmod(sock_file, 0o777)
    READY_PATH.touch()

    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                buf = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                resp = _handle(rag, json.loads(buf.decode("utf-8")))
            except Exception:
                traceback.print_exc()
                resp = {"answer": "", "citations": [], "confidence": 0.0}
            conn.sendall(json.dumps(resp, ensure_ascii=False).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
