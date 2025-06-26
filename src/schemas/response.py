from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictResponse(BaseModel):
    """Response schema for engagement prediction."""
    
    engagement_score: float = Field(
        ..., 
        description="Predicted engagement score"
    )
    
    sentiment_score: float = Field(
        ..., 
        description="Computed sentiment score from comments (0-1 range)",
        ge=0.0,
        le=1.0
    )
    
    cleaned_text: str = Field(
        ..., 
        description="Preprocessed text that was used for prediction"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "engagement_score": 0.75,
                "sentiment_score": 0.8,
                "cleaned_text": "check out this amazing sunset beautiful nature"
            }
        }

class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(default="healthy", description="Service health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded successfully")
    device: str = Field(..., description="Device being used (cpu/cuda)")

class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    
    model_info: Dict[str, Any] = Field(..., description="Detailed model information")
    
    class Config:
        schema_extra = {
            "example": {
                "model_info": {
                    "model_type": "CLIPEngagementRegressor",
                    "device": "cuda:0",
                    "model_path": "models/weights/best_model_lora.pth",
                    "uses_lora": True,
                    "parameters": 428000000,
                    "trainable_parameters": 2000000
                }
            }
        } 