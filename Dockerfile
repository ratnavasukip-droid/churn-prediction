FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Make Python find our code
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project & ensure start script is executable
COPY . /app
RUN chmod +x /app/start.sh

# Expose ports (Render will route $PORT to Streamlit)
EXPOSE 8501
EXPOSE 8000

# Start: API in background, Streamlit on $PORT (handled by start.sh)
CMD ["/app/start.sh"]
