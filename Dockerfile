# Multi-stage build for production-ready Gradio app
FROM python:3.11-slim as base

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser -m

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data/slp_vector_db \
    /app/generated_case_files \
    /app/prompts \
    /home/appuser/.cache/whisper \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /home/appuser

# Set environment variables for model caching
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TORCH_HOME=/home/appuser/.cache/torch
ENV XDG_CACHE_HOME=/home/appuser/.cache

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Expose Gradio port
EXPOSE 7860

# Run the Docker-specific application
CMD ["python", "main_docker.py"]
