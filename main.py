#!/usr/bin/env python3
"""
Main entry point for the Social Media Engagement Prediction API.
"""

import uvicorn

if __name__ == "__main__":
    # Import settings inside the main block to ensure proper path resolution
    from src.config import settings
    
    # Use the new get_uvicorn_config method for better configuration
    config = settings.get_uvicorn_config()
    uvicorn.run(
        "src.app:app",
        **config
    )
