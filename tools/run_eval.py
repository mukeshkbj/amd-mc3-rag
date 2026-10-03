#!/usr/bin/env python3
"""Grader-style eval for MC3 against the starter kit's sample-questions.json.

Score rule per the contract: a question scores only when the answer is right
(after normalization) AND the citation set is exactly right — no partial credit.

    python3 tools/run_eval.py \
        --kit ../mc3-starter-kit \
        --corpus <path-to-corpus-as-mounted>   # e.g. mc3-corpus or /app/corpus
        --outdir /tmp/mc3-out \
        [--script app/app.py] [--timeout 30]

Run `setup.sh` in the kit first (the permission case must exist to be tested).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_NORM = re.compile(r"[\s\-\.·_]")


def norm(s: str) -> str:
    return _NORM.sub("", s.upper())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=Path, required=True,
                    help="path to mc3-starter-kit (holds sample-questions.json)")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path(tempfile.mkdtemp()))
    ap.add_argument("--script", default="app/app.py")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    qs = json.loads((args.kit / "sample-questions.json")
                    .read_text(encoding="utf-8"))["queries"]

    # outputs land where the harness looks: MC3_OUTPUT_DIR for app.py,
    # --outdir for this scorer — keep them the same place
    import os
    env = dict(os.environ, MC3_OUTPUT_DIR=str(args.outdir))

    # index once, like the harness
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, args.script, "--index", str(args.corpus)],
        capture_output=True, text=True, timeout=600, env=env)
    print(f"--index rc={proc.returncode} in {time.time()-t0:.1f}s")
    if proc.returncode != 0:
        print((proc.stderr or proc.stdout)[-500:]); return 1

    total = 0
    for q in qs:
        qid = f"query_{q['n']:02d}"
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, args.script, "--corpus", str(args.corpus),
             "--query-id", qid, "--query", q["query"]],
            capture_output=True, text=True, timeout=args.timeout + 60, env=env)
        dt = time.time() - t0
        out = Path(args.outdir) / f"{qid}_output.json"
        if proc.returncode != 0 or not out.exists():
            print(f"Q{q['n']:02d} [FAIL exec] rc={proc.returncode} "
                  f"{(proc.stderr or '')[-160:]}"); continue
        payload = json.loads(out.read_text(encoding="utf-8"))
        got_a = payload.get("answer", "")
        got_c = [c.replace("\\", "/") for c in payload.get("citations", [])]
        want_a, want_c = q["expected_answer"], q["expected_citations"]
        ans_ok = norm(got_a) == norm(want_a)
        cit_ok = sorted(got_c) == sorted(want_c)
        ok = ans_ok and cit_ok
        total += 20 if ok else 0
        tag = "PASS" if ok else ("answer-only" if ans_ok else "FAIL")
        if dt > args.timeout:
            tag = "TIMEOUT-" + tag  # over the 30s per-query budget: scores 0
            total -= 20 if ok else 0
        print(f"Q{q['n']:02d} [{tag}] ({dt:.1f}s) a={got_a!r} c={got_c}")
        if not ans_ok:
            print(f"      want a={want_a!r}")
        if not cit_ok:
            print(f"      want c={want_c}")

    print(f"\n{total}/200 on the SAMPLE set (graded set differs)")
    return 0 if total == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
