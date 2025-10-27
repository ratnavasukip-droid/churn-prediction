FROM python:3.10-slim

# System deps (faster builds + wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Pre-generate data & model at build time (bakes artifacts into image)
RUN python data/generate_synthetic.py && \
    python src/train.py

# Expose Streamlit default and API port (Render only needs the $PORT one)
EXPOSE 8501
EXPOSE 8000

# Launch script: runs API (8000) + Streamlit ($PORT)
CMD ["/app/start.sh"]
