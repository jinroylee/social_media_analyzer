# Multi-stage build for minimal image size
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set environment variables for build
ENV UV_CACHE_DIR=/tmp/uv-cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync (create venv and install)
RUN uv sync --frozen --no-dev --no-install-project && \
    # Install CPU-only PyTorch directly (never install CUDA version)
    .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    # Aggressive cleanup of virtual environment
    find /app/.venv -name "*.pyc" -delete && \
    find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -name "*.pyo" -delete && \
    find /app/.venv -name "*.pyd" -delete && \
    find /app/.venv -name "*.so" -exec strip {} \; 2>/dev/null || true && \
    find /app/.venv -name "*.a" -delete && \
    find /app/.venv -name "*.la" -delete && \
    # Remove test files and documentation
    find /app/.venv -path "*/tests/*" -delete 2>/dev/null || true && \
    find /app/.venv -path "*/test/*" -delete 2>/dev/null || true && \
    find /app/.venv -name "*.md" -delete && \
    find /app/.venv -name "*.txt" -delete && \
    find /app/.venv -name "*.rst" -delete && \
    find /app/.venv -name "LICENCE*" -delete && \
    find /app/.venv -name "LICENSE*" -delete && \
    find /app/.venv -name "COPYING*" -delete && \
    find /app/.venv -name "AUTHORS*" -delete && \
    find /app/.venv -name "CONTRIBUTORS*" -delete && \
    find /app/.venv -name "CHANGELOG*" -delete && \
    find /app/.venv -name "NEWS*" -delete && \
    # Remove unnecessary torch files
    find /app/.venv -name "*cuda*" -delete 2>/dev/null || true && \
    find /app/.venv -name "*tensorrt*" -delete 2>/dev/null || true && \
    find /app/.venv -name "*cudnn*" -delete 2>/dev/null || true && \
    # Remove other heavy ML libraries if not needed
    find /app/.venv -path "*/site-packages/torch/test*" -delete 2>/dev/null || true && \
    find /app/.venv -path "*/site-packages/torchvision/datasets*" -delete 2>/dev/null || true && \
    # Clear UV cache
    rm -rf /tmp/uv-cache

# Production stage - minimal runtime image
FROM python:3.11-slim as production

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install only essential runtime dependencies including BLAS libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    curl \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy only the cleaned virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Set environment variables - CRITICAL: Set these before copying app code
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    HOST=0.0.0.0 \
    PORT=8000 \
    RELOAD=false \
    LOG_LEVEL=info \
    FORCE_CPU=true

# Copy application code (minimal necessary files)
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser modelfactory/ ./modelfactory/

# Create models directory
RUN mkdir -p /app/models && chown appuser:appuser /app/models

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application using the virtual environment's Python
CMD ["/app/.venv/bin/python", "main.py"] 