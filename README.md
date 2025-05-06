# Social Media Engagement Prediction Microservice

A FastAPI-based microservice for predicting engagement on social media posts using AI models.

## Overview

This project provides a complete pipeline for social media engagement prediction:

1. **Data Preprocessing**: Clean and extract features from social media data
2. **Model Training**: Train both baseline (LightGBM) and deep learning (CLIP-based) models
3. **Serving API**: Deploy models as a FastAPI-based API for real-time predictions

The solution leverages multimodal data (both images and text) to predict engagement metrics on social media posts. The deep learning approach uses OpenAI's CLIP model for understanding both visual and textual content.

## Project Structure

```
social_media_analyze_microserver/
├── data_preprocessing/
│   ├── preprocess.py          # Script to clean data and extract features
│   └── __init__.py
├── models/
│   ├── clip_regressor.py      # Definition of CLIPEngagementRegressor model and head
│   ├── __init__.py
│   └── lightgbm_model.txt     # (Optional) saved LightGBM model or its parameters
├── training/
│   ├── train_lightgbm.py      # Trains and evaluates the LightGBM baseline
│   ├── train_torch.py         # Trains the PyTorch end-to-end model
│   ├── evaluate_torch.py      # (Optional) evaluation script for the PyTorch model
│   └── __init__.py
├── serving/
│   ├── app.py                 # FastAPI application
│   ├── predictor.py           # Helper for model loading and inference (used by app.py)
│   └── __init__.py
├── utils/
│   ├── text_processing.py     # Functions for text cleaning, emoji removal, etc.
│   ├── metrics.py             # Functions for computing MAE, Spearman, etc.
│   ├── config.py              # Configuration (hyperparameters, file paths)
│   └── __init__.py
├── runpod/
│   ├── launch_training.sh     # Shell script to set up env and run training on Runpod
│   ├── launch_server.sh       # Script to launch FastAPI server on Runpod
│   └── __init__.py
├── requirements.txt           # Python dependencies for pip
└── README.md                  # Instructions and project overview
```

## Installation

### Prerequisites

- Python 3.9+
- pip for package installation
- (Optional) CUDA-compatible GPU for faster training

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/social_media_analyze_microserver.git
   cd social_media_analyze_microserver
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```bash
   mkdir -p data/processed
   mkdir -p data/images
   mkdir -p models/saved
   mkdir -p results
   ```

## Usage

### Data Preprocessing

1. Place your raw social media data in `data/raw_data.csv`
2. Place images in `data/images/`
3. Run the preprocessing script:
   ```bash
   python data_preprocessing/preprocess.py
   ```

### Training Models

1. To train the LightGBM baseline model:
   ```bash
   python training/train_lightgbm.py
   ```

2. To train the PyTorch CLIP-based model:
   ```bash
   python training/train_torch.py
   ```

3. To evaluate the trained models:
   ```bash
   python training/evaluate_torch.py
   ```

### Running the API Server

1. Start the FastAPI server:
   ```bash
   cd serving
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Access the API documentation at http://localhost:8000/docs

### Deploying on Runpod

1. To run training on Runpod:
   ```bash
   bash runpod/launch_training.sh
   ```

2. To deploy the server on Runpod:
   ```bash
   bash runpod/launch_server.sh
   ```

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /predict`: Make predictions with JSON input
- `POST /predict_upload`: Make predictions with multipart form data (file upload)
- `POST /predict_batch`: Make batch predictions
- `GET /model_info`: Get information about the loaded model

## Model Architecture

### LightGBM Baseline

The LightGBM model uses text-based features extracted from social media posts to predict engagement. This serves as a baseline for comparison with the deep learning approach.

### CLIP-Based Deep Learning Model

The deep learning model uses OpenAI's CLIP architecture to understand both images and text in social media posts:

1. CLIP processes both the image and text, generating embeddings
2. A custom regression head predicts the engagement score from these embeddings
3. The model can be fine-tuned on domain-specific data

## Performance

The models are evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Spearman Rank Correlation
- R^2 Score

Typically, the CLIP-based model outperforms the LightGBM baseline due to its ability to understand both visual and textual content.

## License

[MIT License](LICENSE)

## Contributors

- Your Name (@yourusername)

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [PyTorch](https://pytorch.org/) 