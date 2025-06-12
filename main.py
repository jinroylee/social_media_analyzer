#!/usr/bin/env python3
"""
Main entry point for the Social Media Engagement Prediction API.
"""

import uvicorn
from src.config import settings

if __name__ == "__main__":
    # Use the new get_uvicorn_config method for better configuration
    config = settings.get_uvicorn_config()
    uvicorn.run(
        "src.app:app",
        **config
    )
