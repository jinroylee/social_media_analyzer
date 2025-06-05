from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import base64
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPTokenizer, pipeline
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_regressor import CLIPEngagementRegressor

app = FastAPI()

# Load models and processors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Load the LoRA-enabled model
model = CLIPEngagementRegressor(use_lora=True, lora_rank=8)
try:
    model.load_state_dict(torch.load('models/best_model_lora.pth', map_location=device))
    print("Loaded LoRA model successfully")
except FileNotFoundError:
    print("LoRA model not found, using base model")
    # Fallback to regular model if LoRA model doesn't exist
    model = CLIPEngagementRegressor(use_lora=False)
    try:
        model.load_state_dict(torch.load('models/clip_engagement_model.pt', map_location=device))
        print("Loaded base model successfully")
    except FileNotFoundError:
        print("No trained model found, using untrained model")

model.to(device)
model.eval()

# Text cleaning patterns
emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          "]+", flags=re.UNICODE)

url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def clean_text(text):
    """Clean text by removing URLs, emojis, and normalizing"""
    text = text.lower()
    text = url_pattern.sub('', text)
    text = emoji_pattern.sub('', text)
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

class PredictRequest(BaseModel):
    image_base64: str
    text: str

class PredictResponse(BaseModel):
    engagement_score: float
    sentiment_score: float

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # Decode image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Clean and process text
        cleaned_text = clean_text(request.text)
        
        # Get sentiment
        sentiment_result = sentiment_pipeline(cleaned_text)[0]
        sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'LABEL_2' else -sentiment_result['score']
        
        # Process inputs for model
        image_inputs = clip_processor(images=image, return_tensors="pt")
        text_inputs = clip_tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        # Move to device
        pixel_values = image_inputs['pixel_values'].to(device)
        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs['attention_mask'].to(device)
        sentiment_tensor = torch.tensor([[sentiment_score]], dtype=torch.float32).to(device)
        
        # Make prediction
        with torch.no_grad():
            engagement_score = model(pixel_values, input_ids, attention_mask, sentiment_tensor)
            engagement_score = engagement_score.item()
        
        return PredictResponse(
            engagement_score=engagement_score,
            sentiment_score=sentiment_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)