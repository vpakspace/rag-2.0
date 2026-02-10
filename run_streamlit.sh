#!/usr/bin/env bash
# Run RAG 2.0 Streamlit UI on port 8502
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

exec streamlit run streamlit_app.py --server.port 8502
