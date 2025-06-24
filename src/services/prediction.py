import torch
import os
import boto3
from botocore.exceptions import ClientError
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
        self.s3_client = None
        
        
        # Initialize S3 client if needed
        if settings.USE_S3_MODEL:
            try:
                self.s3_client = boto3.client('s3', region_name=settings.AWS_REGION)
                print("S3 client initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize S3 client: {e}")
                print("Falling back to local model loading")
        
        self._load_model()
    
    def _download_model_from_s3(self) -> str:
        """Download model from S3 and return local path."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        try:
            # Create local directory if it doesn't exist
            local_dir = os.path.dirname(settings.S3_LOCAL_MODEL_PATH)
            os.makedirs(local_dir, exist_ok=True)
            
            print(f"Downloading model from s3://{settings.S3_BUCKET_NAME}/{settings.S3_MODEL_KEY}")
            
            # Download model from S3
            self.s3_client.download_file(
                settings.S3_BUCKET_NAME,
                settings.S3_MODEL_KEY,
                settings.S3_LOCAL_MODEL_PATH
            )
            
            print(f"Successfully downloaded model to {settings.S3_LOCAL_MODEL_PATH}")
            return settings.S3_LOCAL_MODEL_PATH
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"Model not found in S3: s3://{settings.S3_BUCKET_NAME}/{settings.S3_MODEL_KEY}")
            elif error_code == 'NoSuchBucket':
                raise FileNotFoundError(f"S3 bucket not found: {settings.S3_BUCKET_NAME}")
            else:
                raise RuntimeError(f"S3 error downloading model: {e}")
        except Exception as e:
            raise RuntimeError(f"Error downloading model from S3: {e}")
    
    def _find_model_path(self) -> str:
        """Find the model file in the expected locations."""
        # If S3 is enabled, try to download from S3 first
        if settings.USE_S3_MODEL:
            try:
                return self._download_model_from_s3()
            except Exception as e:
                print(f"Failed to download model from S3: {e}")
                print("Falling back to local model search")
        
        # Try local paths
        possible_paths = [
            self.model_path,
            'modelfactory/models/best_model_lora.pth',
            'models/best_model_lora.pth',
            'best_model_lora.pth'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find model file. Please ensure best_model_lora.pth exists locally or configure S3 settings.")

    def _load_model(self):
        """Load the trained model."""
        try:
            # Find the model path (local or download from S3)
            model_path = self._find_model_path()
            
            # Try to load LoRA model first if enabled
            if settings.USE_LORA:
                self.model = CLIPEngagementRegressor(use_lora=True, lora_rank=settings.LORA_RANK)
            else:
                self.model = CLIPEngagementRegressor(use_lora=False)
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
            
        except FileNotFoundError as e:
            print(f"Model not found: {e}")
            print("Trying fallback options...")
            try:
                # Fallback to regular model
                self.model = CLIPEngagementRegressor(use_lora=False)
                
                # Try different model names
                fallback_paths = []
                if settings.USE_S3_MODEL:
                    # Try alternative S3 paths
                    fallback_s3_keys = [
                        "models/clip_engagement_model.pt",
                        "models/local/best_model_lora.pth"
                    ]
                    for key in fallback_s3_keys:
                        try:
                            original_key = settings.S3_MODEL_KEY
                            settings.S3_MODEL_KEY = key
                            fallback_path = self._download_model_from_s3()
                            settings.S3_MODEL_KEY = original_key  # Restore original
                            fallback_paths.append(fallback_path)
                            break
                        except:
                            settings.S3_MODEL_KEY = original_key  # Restore original
                            continue
                
                # Add local fallback paths
                fallback_paths.extend([
                    self.model_path.replace('best_model_lora.pth', 'clip_engagement_model.pt'),
                    'models/clip_engagement_model.pt',
                    'modelfactory/models/clip_engagement_model.pt'
                ])
                
                # Try each fallback path
                model_loaded = False
                for path in fallback_paths:
                    if os.path.exists(path):
                        self.model.load_state_dict(torch.load(path, map_location=self.device))
                        print(f"Loaded fallback model from {path}")
                        model_loaded = True
                        break
                
                if not model_loaded:
                    print("No trained model found, using untrained model")
                    self.model = CLIPEngagementRegressor(use_lora=False)
                    
            except Exception as fallback_error:
                print(f"Error loading fallback model: {fallback_error}")
                print("Using untrained model")
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
                    image=model_inputs['pixel_values'],
                    input_ids=model_inputs['input_ids'],
                    attention_mask=model_inputs['attention_mask'],
                    sentiment=model_inputs['sentiment_tensor']
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
            "uses_s3": settings.USE_S3_MODEL,
            "s3_bucket": settings.S3_BUCKET_NAME if settings.USE_S3_MODEL else None,
            "s3_key": settings.S3_MODEL_KEY if settings.USE_S3_MODEL else None,
            "uses_lora": hasattr(self.model.clip_model, 'peft_config'),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        } 