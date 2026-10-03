# MC3 — RAG on the mandated ROCm base. Layer identity is checked; never squash.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# --no-deps everywhere: a stray dep can silently replace ROCm torch with a CUDA
# build (verified immediately after install so the build fails loudly).
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt && \
    python3 -c "import torch, transformers, huggingface_hub, httpx2, fitz, PIL; assert 'rocm' in torch.__version__, torch.__version__"

# NO outbound network at eval — weights ride inside the image (~17 GiB).
#   huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3-vl-8b
COPY --from=model /qwen3-vl-8b/ /models/qwen3-vl-8b/
RUN test -s /models/qwen3-vl-8b/config.json
ENV MC3_MODEL_DIR=/models/qwen3-vl-8b
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY app/ /app/
RUN mkdir -p /app/corpus /app/output /app/index

# Daemon loads the model at container start (inside the 10-min startup budget);
# the --index exec and the ten --query execs are then thin socket clients.
CMD ["sh", "-c", "python3 /app/daemon.py >> /tmp/mc3-daemon.log 2>&1 & exec sleep infinity"]
