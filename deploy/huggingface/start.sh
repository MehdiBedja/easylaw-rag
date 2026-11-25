#!/usr/bin/env bash
# Start-up of the Space: the web page comes up at once (it shows "starting" until everything is ready) while the
# index is downloaded and unpacked into Qdrant in the background.
#
# The index dataset holds one tar per collection (the collection's Qdrant storage folder, same Qdrant version)
# and processed/ (text id -> citation lookup).
set -uo pipefail

DATA="${DATA_DIR:-/home/user/data}"
INDEX_REPO="${INDEX_REPO:?set INDEX_REPO (the Hugging Face dataset holding the index) in the Space variables}"
mkdir -p "$DATA/storage"

(
  set -e
  echo "[start] downloading index from $INDEX_REPO"
  python -c "from huggingface_hub import snapshot_download as d; d('$INDEX_REPO', repo_type='dataset', local_dir='$DATA/index')"
  rm -rf "$HOME/app/data/processed" && mkdir -p "$HOME/app/data" && cp -r "$DATA/index/processed" "$HOME/app/data/processed"
  for t in "$DATA"/index/*.tar; do
    echo "[start] unpacking $(basename "$t")"
    tar -xf "$t" -C "$DATA/storage" && rm -f "$t"   # free the disk as we go
  done
  echo "[start] starting Qdrant"
  cd "$DATA"
  export QDRANT__STORAGE__STORAGE_PATH="$DATA/storage" QDRANT__STORAGE__SNAPSHOTS_PATH="$DATA/snapshots" \
         QDRANT__TELEMETRY_DISABLED=true QDRANT__SERVICE__HOST=127.0.0.1
  exec qdrant
) &

cd "$HOME/app"
exec python -m easylaw_rag.app.server
