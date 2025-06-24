# Multi-stage build for efficient Docker image
FROM --platform=linux/amd64 python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory for build
WORKDIR /app

# Copy uv configuration files first for better caching
COPY pyproject.toml uv.lock ./

# Copy README.md and source files needed for package build
COPY README.md ./
COPY src/ ./src/
COPY __init__.py ./
COPY modelfactory/ ./modelfactory/

# Install dependencies using uv, then replace PyTorch with CPU-only version
RUN uv sync --frozen --no-dev && \
    # Remove CUDA PyTorch and install CPU-only version to save ~1GB
    /app/.venv/bin/pip uninstall -y torch torchvision && \
    /app/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Clean up build artifacts to reduce layer size
RUN find /app/.venv -name "*.pyc" -delete && \
    find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -name "*.pyo" -delete && \
    find /app/.venv -name "*.pyd" -delete && \
    find /app/.venv -name "*.so" -exec strip {} \; 2>/dev/null || true

# Production stage - use even smaller base image
FROM --platform=linux/amd64 python:3.11-slim as production

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install only essential runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Install uv in production stage
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy the virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8000 \
    RELOAD=false \
    LOG_LEVEL=info \
    FORCE_CPU=true \
    PIP_NO_CACHE_DIR=1

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code (only necessary files)
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser modelfactory/ ./modelfactory/

# Create necessary directories for models
RUN mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["python", "main.py"] 