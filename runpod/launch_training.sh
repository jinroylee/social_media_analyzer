#!/bin/bash
# Shell script to set up environment and run training on Runpod

# Exit on any error
set -e

# Print commands before executing
set -x

# Configuration
PYTHON_VERSION="3.9"
WORKDIR="/workspace"
REPOSITORY_URL="https://github.com/yourusername/social_media_analyze_microserver.git"
BRANCH="main"

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

# Download data if needed
if [ ! -f "data/processed/train_data.csv" ]; then
    echo "Downloading and preprocessing data..."
    # Add your data download and preprocessing commands here
    # For example:
    # wget -O data/raw_data.zip https://example.com/dataset.zip
    # unzip data/raw_data.zip -d data/
    # python data_preprocessing/preprocess.py
fi

# Run training for LightGBM baseline
echo "Training LightGBM baseline model..."
python training/train_lightgbm.py

# Run training for PyTorch model
echo "Training PyTorch CLIP-based model..."
python training/train_torch.py

# Run evaluation
echo "Evaluating models..."
python training/evaluate_torch.py

echo "Training complete!" 