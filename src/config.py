"""
Configuration settings for the Social Media Engagement Prediction API.
"""

import os
from typing import List

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
    MODEL_PATH: str = os.getenv("MODEL_PATH", "modelfactory/models/best_model_lora.pth")
    USE_LORA: bool = os.getenv("USE_LORA", "true").lower() == "true"
    LORA_RANK: int = int(os.getenv("LORA_RANK", "8"))
    
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

# Create global settings instance
settings = Settings() 