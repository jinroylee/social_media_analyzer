"""
FastAPI application for serving social media engagement prediction model.
"""
import os
import io
import base64
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import numpy as np

from serving.predictor import ModelPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Social Media Engagement Predictor",
    description="API for predicting engagement on social media posts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model request/response schemas
class PredictionRequest(BaseModel):
    caption: str = Field(..., description="The caption or text of the social media post")
    image_b64: Optional[str] = Field(None, description="Base64 encoded image data")

class PredictionResponse(BaseModel):
    engagement_score: float = Field(..., description="Predicted engagement score")
    confidence: Optional[float] = Field(None, description="Confidence score for the prediction")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores if available")

# Dependency for the model predictor
def get_predictor():
    return ModelPredictor()

# Healthcheck endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Prediction endpoint for JSON
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, predictor: ModelPredictor = Depends(get_predictor)):
    try:
        # Decode image if provided
        image = None
        if request.image_b64:
            image_data = base64.b64decode(request.image_b64)
            image = Image.open(io.BytesIO(image_data))
        
        # Make prediction
        prediction = predictor.predict(request.caption, image)
        
        return PredictionResponse(
            engagement_score=prediction["engagement_score"],
            confidence=prediction.get("confidence"),
            feature_importance=prediction.get("feature_importance")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Prediction endpoint for multipart form data (file upload)
@app.post("/predict_upload", response_model=PredictionResponse)
async def predict_upload(
    caption: str = Form(...),
    image: Optional[UploadFile] = File(None),
    predictor: ModelPredictor = Depends(get_predictor)
):
    try:
        # Process image if provided
        img = None
        if image:
            img_data = await image.read()
            img = Image.open(io.BytesIO(img_data))
        
        # Make prediction
        prediction = predictor.predict(caption, img)
        
        return PredictionResponse(
            engagement_score=prediction["engagement_score"],
            confidence=prediction.get("confidence"),
            feature_importance=prediction.get("feature_importance")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[PredictionRequest], predictor: ModelPredictor = Depends(get_predictor)):
    results = []
    
    for request in requests:
        try:
            # Decode image if provided
            image = None
            if request.image_b64:
                image_data = base64.b64decode(request.image_b64)
                image = Image.open(io.BytesIO(image_data))
            
            # Make prediction
            prediction = predictor.predict(request.caption, image)
            
            results.append(PredictionResponse(
                engagement_score=prediction["engagement_score"],
                confidence=prediction.get("confidence"),
                feature_importance=prediction.get("feature_importance")
            ))
        except Exception as e:
            # Skip failed predictions in batch mode
            results.append(PredictionResponse(
                engagement_score=0.0,
                confidence=0.0,
                feature_importance={"error": str(e)}
            ))
    
    return results

# Model info endpoint
@app.get("/model_info")
def model_info(predictor: ModelPredictor = Depends(get_predictor)):
    return predictor.get_model_info()

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 