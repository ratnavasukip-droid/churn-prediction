#!/usr/bin/env bash
set -e
export PYTHONPATH=/app
export PORT="${PORT:-8501}"

# make sure artifacts exist
python data/generate_synthetic.py
python src/train.py

# start API (port 8000) in background
uvicorn api.app:app --host 0.0.0.0 --port 8000 &

# start Streamlit on public $PORT
streamlit run dashboard/app_streamlit.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT"
