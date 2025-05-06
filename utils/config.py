"""
Configuration file for hyperparameters, file paths, and other settings.
"""
import os
from typing import Dict, Any

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# LightGBM model parameters
LIGHTGBM_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_iterations': 1000,
    'early_stopping_rounds': 50
}

# PyTorch model parameters
TORCH_TRAINING_PARAMS = {
    'clip_model_name': 'openai/clip-vit-base-patch32',
    'freeze_clip': True,
    'batch_size': 32,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'image_dir': os.path.join(DATA_DIR, "images"),
    'image_col': 'image_filename',
    'text_col': 'caption',
    'target_col': 'engagement',
    'save_dir': MODELS_DIR
}

# Model paths
MODEL_PATHS = {
    'torch': os.path.join(MODELS_DIR, 'best_clip_regressor.pt'),
    'lightgbm': os.path.join(MODELS_DIR, 'lightgbm_engagement_predictor.txt')
}

# Model configuration
MODEL_CONFIG = {
    'clip_model_name': 'openai/clip-vit-base-patch32',
    'default_model_type': 'torch'  # 'torch' or 'lightgbm'
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'workers': 4,
    'timeout': 60
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
} 