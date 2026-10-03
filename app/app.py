#!/usr/bin/env python3
"""MC3 grader entrypoint — keeps the starter plumbing, talks to the daemon.

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

Writes /app/output/<query-id>_output.json
    -> {"answer": "...", "citations": [...], "confidence": ...}
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

READY_PATH = Path(os.environ.get("MC3_READY", "/tmp/mc3-rag.ready"))
OUTPUT_DIR = Path(os.environ.get("MC3_OUTPUT_DIR", "/app/output"))
READY_TIMEOUT_S = float(os.environ.get("MC3_READY_TIMEOUT", "540"))


def _addr():
    raw = os.environ.get("MC3_SOCK", "/tmp/mc3-rag.sock")
    if raw.startswith("tcp:"):
        host, port = raw[4:].rsplit(":", 1)
        return socket.AF_INET, (host, int(port))
    return socket.AF_UNIX, raw


def _spawn_daemon() -> None:
    daemon = Path(__file__).resolve().parent / "daemon.py"
    subprocess.Popen(
        [sys.executable, str(daemon)],
        stdout=open("/tmp/mc3-daemon.log", "ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _daemon_loading() -> bool:
    """True if a daemon process is alive (still loading the model). The CMD in
    the image starts one; on the pod we start it by hand. Only spawn our own
    when no daemon exists at all — otherwise two models load into VRAM."""
    try:
        pid = int(Path(str(READY_PATH) + ".loading").read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _wait_ready() -> None:
    deadline = time.time() + READY_TIMEOUT_S
    spawned = False
    while time.time() < deadline:
        if READY_PATH.exists():
            return
        if not spawned and not _daemon_loading():
            _spawn_daemon()
            spawned = True
        time.sleep(0.5)
    raise TimeoutError("MC3 daemon never became ready")


def _rpc(req: dict) -> dict:
    family, addr = _addr()
    s = socket.socket(family, socket.SOCK_STREAM)
    s.connect(addr)
    s.sendall(json.dumps(req).encode("utf-8"))
    s.shutdown(socket.SHUT_WR)
    buf = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    args = ap.parse_args()

    _wait_ready()

    if args.index is not None:
        _rpc({"cmd": "index", "corpus": str(args.index)})
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    resp = _rpc({"cmd": "query", "corpus": str(args.corpus), "query": args.query})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{args.query_id}_output.json"
    out.write_text(
        json.dumps(
            {
                "answer": str(resp.get("answer", "")),
                "citations": list(resp.get("citations", [])),
                "confidence": float(resp.get("confidence", 0.0)),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
