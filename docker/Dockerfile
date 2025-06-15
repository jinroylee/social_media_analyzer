# Multi-stage build for efficient Docker image
FROM python:3.11-slim as builder

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
    UV_CACHE_DIR=/tmp/uv-cache

# Set working directory for build
WORKDIR /app

# Copy uv configuration files first for better caching
COPY pyproject.toml uv.lock ./

# Copy README.md and source files needed for package build
COPY README.md ./
COPY src/ ./src/
COPY __init__.py ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.11-slim as production

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

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
    FORCE_CPU=false

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories and set permissions
RUN mkdir -p /app/modelfactory/models && \
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