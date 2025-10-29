#!/usr/bin/env bash
set -e

export PYTHONPATH=/app
export PORT="${PORT:-8501}"

# Ensure artifacts exist
python data/generate_synthetic.py
python src/train.py

# Start API
uvicorn api.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "[start.sh] Launched API pid=$API_PID"

# Wait until API responds (max ~30s)
for i in {1..30}; do
  if curl -sf http://127.0.0.1:8000/ping >/dev/null 2>&1 || curl -sf http://127.0.0.1:8000/docs >/dev/null 2>&1; then
    echo "[start.sh] API is up ✅"
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "[start.sh] API crashed ❌"
    exit 1
  fi
  echo "[start.sh] Waiting for API..."
  sleep 1
done

# Start Streamlit on public $PORT
exec streamlit run dashboard/app_streamlit.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT"
