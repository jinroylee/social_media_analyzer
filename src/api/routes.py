from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

from ..schemas.request import PredictRequest
from ..schemas.response import PredictResponse, HealthResponse, ModelInfoResponse
from ..services.preprocessing import PreprocessingService
from ..services.prediction import PredictionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global services (will be initialized on startup)
preprocessing_service: PreprocessingService = None
prediction_service: PredictionService = None

def get_preprocessing_service() -> PreprocessingService:
    """Dependency to get preprocessing service."""
    global preprocessing_service
    if preprocessing_service is None:
        preprocessing_service = PreprocessingService()
    return preprocessing_service

def get_prediction_service() -> PredictionService:
    """Dependency to get prediction service."""
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService()
    return prediction_service

@router.post("/predict", response_model=PredictResponse)
async def predict_engagement(
    request: PredictRequest,
    preprocessing_svc: PreprocessingService = Depends(get_preprocessing_service),
    prediction_svc: PredictionService = Depends(get_prediction_service)
):
    """
    Predict engagement score for social media content.
    
    This endpoint accepts raw text, a base64-encoded image, and comments,
    then returns an engagement score prediction along with sentiment analysis.
    """
    try:
        logger.info(f"Processing prediction request for text: {request.raw_text[:50]}...")
        
        # Preprocess the request
        model_inputs = preprocessing_svc.preprocess_request(
            raw_text=request.raw_text,
            image_base64=request.image_base64,
            comments=request.comments,
            device=prediction_svc.device
        )
        
        # Make prediction
        engagement_score = prediction_svc.predict(model_inputs)
        
        logger.info(f"Prediction completed. Engagement score: {engagement_score}")
        
        return PredictResponse(
            engagement_score=engagement_score,
            sentiment_score=model_inputs['sentiment_score'],
            cleaned_text=model_inputs['cleaned_text']
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except RuntimeError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health", response_model=HealthResponse)
async def health_check(
    prediction_svc: PredictionService = Depends(get_prediction_service)
):
    """
    Health check endpoint to verify service status.
    """
    try:
        model_loaded = prediction_svc.model is not None
        device = str(prediction_svc.device)
        
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            device=device
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown"
        )

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(
    prediction_svc: PredictionService = Depends(get_prediction_service)
):
    """
    Get detailed information about the loaded model.
    """
    try:
        model_info = prediction_svc.get_model_info()
        return ModelInfoResponse(model_info=model_info)
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.get("/")
async def root():
    """
    Root endpoint with basic API information.
    """
    return {
        "name": "Social Media Engagement Prediction API",
        "version": "1.0.0",
        "description": "API for predicting social media engagement using CLIP and sentiment analysis",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    } 