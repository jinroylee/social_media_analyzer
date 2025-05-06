"""
Definition of CLIPEngagementRegressor model and regression head.
"""
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from typing import Dict, Union, Tuple

class RegressionHead(nn.Module):
    """
    Regression head that takes CLIP embeddings and outputs engagement predictions.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class CLIPEngagementRegressor(nn.Module):
    """
    End-to-end model that uses CLIP embeddings to predict social media engagement.
    """
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", freeze_clip: bool = True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Freeze CLIP parameters if specified
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
                
        # Get embedding dimension from CLIP
        clip_embedding_dim = self.clip.config.projection_dim
        
        # Create regression head
        self.regression_head = RegressionHead(input_dim=clip_embedding_dim)
        
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, 
               attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
            pixel_values: Tensor containing the image features
            input_ids: Tensor containing the text token ids
            attention_mask: Tensor containing the text attention mask
            
        Returns:
            Tensor containing the predicted engagement value
        """
        # Get CLIP embeddings
        clip_outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True
        )
        
        # Use combined multimodal embedding from CLIP
        combined_embedding = clip_outputs.pooler_output
        
        # Pass through regression head
        engagement_prediction = self.regression_head(combined_embedding)
        
        return engagement_prediction
    
    def process_inputs(self, images, texts) -> Dict[str, torch.Tensor]:
        """
        Process raw images and texts using the CLIP processor.
        
        Args:
            images: List of PIL images or image paths
            texts: List of text strings
            
        Returns:
            Dictionary containing processed inputs for the model
        """
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        return inputs 