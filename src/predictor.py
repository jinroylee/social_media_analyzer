"""
Helper for model loading and inference.
"""
import os
import torch
import lightgbm as lgb
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Union, Optional
import logging

from models.clip_regressor import CLIPEngagementRegressor
from utils.text_processing import clean_text, remove_emojis
from utils.config import MODEL_PATHS, MODEL_CONFIG

logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Helper class for loading models and making predictions.
    """
    def __init__(self, model_type: str = "torch"):
        """
        Initialize the model predictor.
        
        Args:
            model_type: Type of model to use ('torch' or 'lightgbm')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        
        # Load the appropriate model
        self._load_model()
    
    def _load_model(self):
        """
        Load the model based on the model type.
        """
        try:
            if self.model_type == "torch":
                # Load PyTorch model
                logger.info(f"Loading PyTorch model from {MODEL_PATHS['torch']}")
                self.model = CLIPEngagementRegressor(
                    clip_model_name=MODEL_CONFIG['clip_model_name'],
                    freeze_clip=True
                )
                self.model.load_state_dict(torch.load(MODEL_PATHS['torch']))
                self.model.eval()
                
                # Move to GPU if available
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(self.device)
                
            elif self.model_type == "lightgbm":
                # Load LightGBM model
                logger.info(f"Loading LightGBM model from {MODEL_PATHS['lightgbm']}")
                self.model = lgb.Booster(model_file=MODEL_PATHS['lightgbm'])
                self.feature_names = self.model.feature_name()
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            logger.info(f"Successfully loaded {self.model_type} model")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, text: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        
        Args:
            text: Text/caption of the social media post
            image: Image for the social media post (optional)
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if self.model_type == "torch":
                return self._predict_torch(text, image)
            elif self.model_type == "lightgbm":
                return self._predict_lightgbm(text, image)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _predict_torch(self, text: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Make a prediction with the PyTorch model.
        
        Args:
            text: Text/caption of the social media post
            image: Image for the social media post
            
        Returns:
            Dictionary containing prediction results
        """
        # Check if image is provided
        if image is None:
            # Use a blank image if none provided
            image = Image.new('RGB', (224, 224), color='white')
        
        # Process inputs
        inputs = self.model.process_inputs([image], [text])
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        # Get prediction value
        prediction = outputs.squeeze().cpu().item()
        
        return {
            "engagement_score": prediction,
            "confidence": None,  # Model doesn't provide confidence
            "model_type": "torch"
        }
    
    def _predict_lightgbm(self, text: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Make a prediction with the LightGBM model.
        
        Args:
            text: Text/caption of the social media post
            image: Image for the social media post (not used for LightGBM)
            
        Returns:
            Dictionary containing prediction results
        """
        # Process text features
        cleaned_text = clean_text(text)
        no_emoji = remove_emojis(text)
        
        # Extract features
        features = self._extract_lightgbm_features(text, cleaned_text, no_emoji)
        
        # Make prediction
        prediction = self.model.predict([features])[0]
        
        # Get feature importance if available
        feature_importance = None
        if self.feature_names is not None:
            importance = self.model.feature_importance()
            feature_importance = dict(zip(self.feature_names, importance))
        
        return {
            "engagement_score": prediction,
            "confidence": None,  # LightGBM doesn't provide confidence by default
            "feature_importance": feature_importance,
            "model_type": "lightgbm"
        }
    
    def _extract_lightgbm_features(self, text: str, cleaned_text: str, no_emoji: str) -> List[float]:
        """
        Extract features for the LightGBM model.
        
        Args:
            text: Original text
            cleaned_text: Cleaned text
            no_emoji: Text with emojis removed
            
        Returns:
            List of feature values
        """
        # This is a placeholder - actual feature extraction would depend on your model
        features = [
            len(text),  # Text length
            len(cleaned_text),  # Cleaned text length
            len(no_emoji),  # Text without emojis length
            text.count(' ') + 1,  # Word count
            len(text) - len(no_emoji),  # Emoji count
            # Add more features as needed
        ]
        
        return features
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            "model_type": self.model_type,
            "model_path": MODEL_PATHS[self.model_type],
        }
        
        if self.model_type == "torch":
            info.update({
                "clip_model": MODEL_CONFIG['clip_model_name'],
                "device": str(self.device)
            })
        elif self.model_type == "lightgbm":
            info.update({
                "num_features": len(self.feature_names) if self.feature_names else 0,
                "feature_names": self.feature_names
            })
        
        return info 