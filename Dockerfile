FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Make Python find our top-level modules
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install deps (layer-cached)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app
RUN chmod +x /app/start.sh

# Pre-generate data & model at build time
RUN python data/generate_synthetic.py && \
    python src/train.py

EXPOSE 8501
EXPOSE 8000

CMD ["/app/start.sh"]
