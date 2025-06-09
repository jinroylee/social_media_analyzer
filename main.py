#!/usr/bin/env python3
"""
Main entry point for the Social Media Engagement Prediction API.
"""

import uvicorn
from src.app import app
from src.config import settings

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL
    )
