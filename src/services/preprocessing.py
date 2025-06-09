import re
import torch
from PIL import Image
from transformers import pipeline, CLIPProcessor, CLIPTokenizer
from typing import List
import io
import base64
from ..config import settings

class PreprocessingService:
    """Service for preprocessing text, images, and comments for the engagement prediction model."""
    
    def __init__(self):
        # Initialize sentiment analysis pipeline
        device_id = 0 if torch.cuda.is_available() and not settings.FORCE_CPU else -1
        self.sentiment_pipeline = pipeline(
            "text-classification", 
            model=settings.SENTIMENT_MODEL_NAME,
            device=device_id,
            batch_size=settings.SENTIMENT_BATCH_SIZE,
            return_all_scores=False
        )
        
        # Initialize CLIP processors
        self.clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(settings.CLIP_MODEL_NAME)
        
        # Text cleaning patterns
        self.emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
        self.url_pattern = re.compile(r"http\S+")
        
        # Sentiment mapping
        self.sentiment_map = {
            "Very Negative": 0, 
            "Negative": 1, 
            "Neutral": 2, 
            "Positive": 3, 
            "Very Positive": 4
        }
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing URLs, emojis, and normalizing whitespace."""
        text = text.lower()
        text = self.url_pattern.sub("", text)
        text = self.emoji_pattern.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def compute_sentiment(self, comments: List[str]) -> float:
        """
        Compute sentiment score from comments.
        Returns normalized sentiment score (0-1 range).
        """
        if len(comments) == 0:
            return 0.5  # Neutral when no comments (2/4 = 0.5)
        
        # Join all comments into a single string for sentiment analysis
        combined_text = " ".join(comments)
        
        try:
            # Get prediction from the model
            result = self.sentiment_pipeline(combined_text)
            
            if 'label' in result[0]:
                label = result[0]['label']
                sentiment_class = self.sentiment_map.get(label, 2)  # Default to neutral
            else:
                sentiment_class = 2  # Default to neutral if 'label' is not found
            
            # Normalize to 0-1 range (0-4 -> 0-1)
            return sentiment_class / 4.0
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return 0.5  # Return neutral on error
    
    def process_image(self, image_base64: str) -> Image.Image:
        """
        Process base64 encoded image and resize to configured size.
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Resize to configured size
            image = image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")
    
    def prepare_model_inputs(self, image: Image.Image, text: str, sentiment_score: float, device: torch.device):
        """
        Prepare inputs for the CLIP engagement model.
        """
        try:
            # Process image for CLIP
            image_inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Process text for CLIP
            text_inputs = self.clip_tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=settings.MAX_TEXT_LENGTH,
                return_tensors="pt"
            )
            
            # Move to device
            pixel_values = image_inputs['pixel_values'].to(device)
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            sentiment_tensor = torch.tensor([[sentiment_score]], dtype=torch.float32).to(device)
            
            return pixel_values, input_ids, attention_mask, sentiment_tensor
            
        except Exception as e:
            raise ValueError(f"Error preparing model inputs: {e}")
    
    def preprocess_request(self, raw_text: str, image_base64: str, comments: List[str], device: torch.device):
        """
        Complete preprocessing pipeline for API request.
        """
        # Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        # Process the image
        image = self.process_image(image_base64)
        
        # Compute sentiment from comments
        sentiment_score = self.compute_sentiment(comments)
        
        # Prepare model inputs
        pixel_values, input_ids, attention_mask, sentiment_tensor = self.prepare_model_inputs(
            image, cleaned_text, sentiment_score, device
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentiment_tensor': sentiment_tensor,
            'cleaned_text': cleaned_text,
            'sentiment_score': sentiment_score
        } 