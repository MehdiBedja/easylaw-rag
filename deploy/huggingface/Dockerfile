# Hugging Face Space (Docker, free CPU tier): Qdrant + the EasyLaw chat app in one container.
# The index (Qdrant snapshots, ~7 GB) is downloaded from a Hugging Face dataset at start-up, see start.sh.
FROM qdrant/qdrant:v1.20.0 AS qdrant

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates libunwind8 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user

# CPU-only PyTorch first (the default wheel bundles CUDA and is several GB larger).
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

# Code from GitHub; the ADD busts the build cache whenever main moves.
ARG REPO=MehdiBedja/easylaw-rag
ADD https://api.github.com/repos/${REPO}/commits/main /tmp/version.json
RUN git clone --depth 1 https://github.com/${REPO}.git app \
    && pip install --no-cache-dir --user -e "./app[app]" huggingface_hub

# Bake both models into the image so a cold start only waits for the index.
RUN python -c "from FlagEmbedding import BGEM3FlagModel, FlagReranker; \
BGEM3FlagModel('BAAI/bge-m3', devices='cpu'); FlagReranker('BAAI/bge-reranker-v2-m3', devices='cpu')"

COPY --chown=user start.sh /home/user/start.sh

# Two CPU cores: rerank fewer, shorter candidates (measured on the laptop: 40 x 512 tokens = ~12 s on a GPU).
ENV EASYLAW_DEVICE=cpu RERANK_CANDIDATES=20 RERANK_MAX_LENGTH=320 RERANK_BATCH=10 \
    HOST=0.0.0.0 PORT=7860 QDRANT_URL=http://127.0.0.1:6333 \
    RATE_LIMIT_PER_MIN=5 RATE_LIMIT_PER_DAY=40
EXPOSE 7860
CMD ["bash", "/home/user/start.sh"]
