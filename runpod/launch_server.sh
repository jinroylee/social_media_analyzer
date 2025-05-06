#!/bin/bash
# Shell script to launch FastAPI server on Runpod

# Exit on any error
set -e

# Print commands before executing
set -x

# Configuration
PYTHON_VERSION="3.9"
WORKDIR="/workspace"
REPOSITORY_URL="https://github.com/yourusername/social_media_analyze_microserver.git"
BRANCH="main"
PORT=8000

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate

# Clone repository if not already present
if [ ! -d "${WORKDIR}/social_media_analyze_microserver" ]; then
    echo "Cloning repository..."
    git clone --branch ${BRANCH} ${REPOSITORY_URL} ${WORKDIR}/social_media_analyze_microserver
else
    echo "Repository already exists, pulling latest changes..."
    cd ${WORKDIR}/social_media_analyze_microserver
    git pull origin ${BRANCH}
fi

# Navigate to project directory
cd ${WORKDIR}/social_media_analyze_microserver

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/processed
mkdir -p models/saved
mkdir -p results

# Download trained models if needed
if [ ! -f "models/saved/best_clip_regressor.pt" ]; then
    echo "Downloading trained models..."
    # Add your model download commands here
    # For example:
    # wget -O models/saved/best_clip_regressor.pt https://example.com/models/best_clip_regressor.pt
    # wget -O models/saved/lightgbm_engagement_predictor.txt https://example.com/models/lightgbm_engagement_predictor.txt
fi

# Start server
echo "Starting FastAPI server..."
cd serving
uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 4 