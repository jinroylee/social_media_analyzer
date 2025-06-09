import torch
import os
from typing import Dict, Any
from modelfactory.models.clip_regressor import CLIPEngagementRegressor
from ..config import settings

class PredictionService:
    """Service for loading the model and making predictions."""
    
    def __init__(self, model_path: str = None):
        # Set device based on configuration
        if settings.FORCE_CPU:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.model_path = model_path or settings.MODEL_PATH
        self._load_model()
    
    def _find_model_path(self) -> str:
        """Find the model file in the expected locations."""
        possible_paths = [
            'modelfactory/models/best_model_lora.pth',
            'models/best_model_lora.pth',
            'best_model_lora.pth'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find model file. Please ensure best_model_lora.pth exists.")
    
    def _load_model(self):
        """Load the trained model."""
        try:
            # Try to load LoRA model first if enabled
            if settings.USE_LORA:
                self.model = CLIPEngagementRegressor(use_lora=True, lora_rank=settings.LORA_RANK)
            else:
                self.model = CLIPEngagementRegressor(use_lora=False)
            
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"Loaded model from {self.model_path}")
            
        except FileNotFoundError:
            print("LoRA model not found, trying base model...")
            try:
                # Fallback to regular model
                self.model = CLIPEngagementRegressor(use_lora=False)
                base_model_path = self.model_path.replace('best_model_lora.pth', 'clip_engagement_model.pt')
                self.model.load_state_dict(torch.load(base_model_path, map_location=self.device))
                print(f"Loaded base model from {base_model_path}")
                
            except FileNotFoundError:
                print("No trained model found, using untrained model")
                self.model = CLIPEngagementRegressor(use_lora=False)
        
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model")
            self.model = CLIPEngagementRegressor(use_lora=False)
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, model_inputs: Dict[str, torch.Tensor]) -> float:
        """
        Make a prediction using the loaded model.
        
        Args:
            model_inputs: Dictionary containing preprocessed inputs
            
        Returns:
            Engagement score prediction
        """
        try:
            with torch.no_grad():
                engagement_score = self.model(
                    model_inputs['pixel_values'],
                    model_inputs['input_ids'],
                    model_inputs['attention_mask'],
                    model_inputs['sentiment_tensor']
                )
                return engagement_score.item()
                
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": "CLIPEngagementRegressor",
            "device": str(self.device),
            "model_path": self.model_path,
            "uses_lora": hasattr(self.model.clip_model, 'peft_config'),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        } 