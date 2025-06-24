"""
Configuration settings for the Social Media Engagement Prediction API.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Get the directory containing this config file
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (since config.py is in src/)
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)
# Path to .env file in project root
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, '.env')

# Load environment variables from .env file with explicit path
load_dotenv(ENV_FILE_PATH)

# Debug: Print if .env file was found
if os.path.exists(ENV_FILE_PATH):
    print(f"✓ Loaded .env file from: {ENV_FILE_PATH}")
else:
    print(f"⚠ .env file not found at: {ENV_FILE_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Config file location: {CONFIG_DIR}")
    print(f"Project root: {PROJECT_ROOT}")

class Settings:
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "Social Media Engagement Prediction API"
    API_DESCRIPTION: str = "API for predicting social media engagement using CLIP and sentiment analysis"
    API_VERSION: str = "1.0.0"
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # Model Settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best_model_lora.pth")
    USE_LORA: bool = os.getenv("USE_LORA", "true").lower() == "true"
    LORA_RANK: int = int(os.getenv("LORA_RANK", "8"))
    
    # S3 Model Settings
    USE_S3_MODEL: bool = os.getenv("USE_S3_MODEL", "false").lower() == "true"
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "socialmediaanalyzer")
    S3_MODEL_KEY: str = os.getenv("S3_MODEL_KEY", "models/best_model_lora.pth")
    S3_LOCAL_MODEL_PATH: str = os.getenv("S3_LOCAL_MODEL_PATH", "models/s3/best_model_lora.pth")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    
    # CLIP Model Settings
    CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14")
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "77"))
    
    # Sentiment Analysis Settings
    SENTIMENT_MODEL_NAME: str = os.getenv("SENTIMENT_MODEL_NAME", "tabularisai/multilingual-sentiment-analysis")
    SENTIMENT_BATCH_SIZE: int = int(os.getenv("SENTIMENT_BATCH_SIZE", "32"))
    
    # Image Processing Settings
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "256"))
    
    # Request Limits
    MAX_TEXT_LENGTH_REQUEST: int = int(os.getenv("MAX_TEXT_LENGTH_REQUEST", "10000"))
    MAX_COMMENTS: int = int(os.getenv("MAX_COMMENTS", "1000"))
    
    # CORS Settings
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Device Settings
    FORCE_CPU: bool = os.getenv("FORCE_CPU", "false").lower() == "true"
    
    def get_uvicorn_config(self) -> Dict[str, Any]:
        """Get uvicorn configuration dictionary."""
        return {
            "host": self.HOST,
            "port": self.PORT,
            "reload": self.RELOAD,
            "log_level": self.LOG_LEVEL.lower()
        }

# Create global settings instance
settings = Settings() 