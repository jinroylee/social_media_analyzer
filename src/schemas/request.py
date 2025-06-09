from pydantic import BaseModel, Field, validator
from typing import List
import base64
from ..config import settings

class PredictRequest(BaseModel):
    """Request schema for engagement prediction."""
    
    raw_text: str = Field(
        ..., 
        description="Raw text content to analyze",
        min_length=1,
        max_length=settings.MAX_TEXT_LENGTH_REQUEST
    )
    
    image_base64: str = Field(
        ..., 
        description=f"Base64 encoded JPEG image ({settings.IMAGE_SIZE}x{settings.IMAGE_SIZE} pixels)"
    )
    
    comments: List[str] = Field(
        default_factory=list,
        description="Array of comment strings for sentiment analysis",
        max_items=settings.MAX_COMMENTS
    )
    
    @validator('image_base64')
    def validate_base64_image(cls, v):
        """Validate that the image is properly base64 encoded."""
        try:
            # Try to decode to validate format
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 encoded image")
    
    @validator('comments')
    def validate_comments(cls, v):
        """Validate comments list."""
        if not isinstance(v, list):
            raise ValueError("Comments must be a list of strings")
        
        # Filter out empty comments and limit length
        filtered_comments = [comment.strip() for comment in v if comment.strip()]
        return filtered_comments[:settings.MAX_COMMENTS]  # Limit to configured max comments
    
    class Config:
        schema_extra = {
            "example": {
                "raw_text": "Check out this amazing sunset! #beautiful #nature",
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                "comments": [
                    "Wow, so beautiful!",
                    "Amazing shot!",
                    "Love this!"
                ]
            }
        } 