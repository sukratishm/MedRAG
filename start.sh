#!/usr/bin/env bash
set -euo pipefail

export DATA_DIR="${DATA_DIR:-/tmp/medrag_data}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export PREFETCH_MODEL="${PREFETCH_MODEL:-1}"

python download_assets.py
exec streamlit run app.py --server.port 7860 --server.address 0.0.0.0
