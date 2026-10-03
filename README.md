# amd-mc3-rag — Mini-Challenge 3 (Lablab × AMD AI Academy)

Retrieval-Augmented Generation over a mixed-filetype corpus, packaged on the
mandated ROCm PyTorch base. A resident **Qwen3-VL-8B-Instruct** daemon loads once
at container start and holds the corpus index in memory; each
`app.py --corpus ... --query-id ... --query ...` exec is a thin socket client, so
per-question work stays well inside the 30-second budget.

Published image: `docker.io/mukeshkbj/amd-mc3-rag:v1`

## Layout

```
app/daemon.py     resident socket server — loads model once, serves index/query
app/app.py        grader entrypoint: --index and --corpus/--query-id/--query
app/indexer.py    corpus walk -> chunks -> persisted index; one bad file never aborts
app/parsers.py    pdf/docx/xlsx/csv/text/source/images; unreadable files are skipped
app/retriever.py  BM25 + identifier expansion + necessary-citation bridge
app/answerer.py   model JSON -> normalized answer + exact-set citations/refusal
tools/run_eval.py starter-kit-style scorer, exact answer+citation match
Dockerfile        rocm/pytorch base + weights baked in (no network at eval)
```

## Behaviour

- Parses `.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt/.log`, `.py`, `.png`, `.jpg`.
- Skips empty directories, unknown types, encrypted PDFs and unreadable files.
- Excludes withdrawn/superseded documents from context.
- Returns `{"answer":"","citations":[],"confidence":...}` when the corpus does
  not answer; never guesses.

## Results

- 200/200 on the official 10-question sample set (AMD GPU pod); graded set differs.
- Image loads the bundled Qwen config/processor with `--network none`.

## Build

```bash
# model weights are NOT in this repo — fetch them first (~17 GiB)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3-vl-8b

docker buildx build --load \
  --build-context model="$(pwd)/models" \
  -t amd-mc3-rag:v1 .

docker run -d --network none --name mc3 amd-mc3-rag:v1
docker cp <corpus>/. mc3:/app/corpus/
docker exec mc3 python3 /app/app.py --index /app/corpus
docker exec mc3 python3 /app/app.py --corpus /app/corpus \
  --query-id query_01 --query "What is the maximum junction temperature?"
```

For local evaluation against the starter kit, download `mc3-starter-kit.zip`
from the challenge page, run `setup.sh`, then point `tools/run_eval.py` at that
kit directory.

Env knobs: `MC3_MODEL_DIR`, `MC3_INDEX_DIR`, `MC3_OUTPUT_DIR`, `MC3_READY`,
`MC3_SOCK` (unix path, or `tcp:host:port` for non-Linux testing), `ALLOW_CPU`
(local testing only; grading requires GPU).
