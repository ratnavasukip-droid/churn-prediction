#!/usr/bin/env bash
set -e

# Ensure model exists (safe to rerun)
python data/generate_synthetic.py
python src/train.py

# Start FastAPI on 8000 in the background
uvicorn api.app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit bound to the platform port
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_PORT=${PORT:-8501}

streamlit run dashboard/app_streamlit.py
