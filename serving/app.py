
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
import base64
import torch
import math
from PIL import Image
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel, pipeline
from models.clip_regressor import CLIPEngagementRegressor

app = FastAPI()

# Load models and processors
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
sentiment_pipeline = pipeline("sentiment-analysis")

model = CLIPEngagementRegressor()
model.load_state_dict(torch.load("models/clip_engagement_model.pt", map_location=torch.device("cpu")))
model.eval()

class PredictRequest(BaseModel):
    thumbnail: str  # base64 encoded
    description: str
    top_comments: Optional[List[str]] = None

class PredictResponse(BaseModel):
    engagement_score: float

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # Decode image
    image_data = base64.b64decode(payload.thumbnail)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image_inputs = clip_processor(images=image, return_tensors="pt")
    img_tensor = image_inputs["pixel_values"]

    # Tokenize text
    full_text = payload.description
    if payload.top_comments:
        full_text += " " + " ".join(payload.top_comments)
    text_inputs = clip_tokenizer(full_text, truncation=True, max_length=77, return_tensors="pt")
    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs["attention_mask"]

    # Sentiment
    sent = sentiment_pipeline(full_text[:512])[0]
    sentiment_score = sent['score'] if sent['label'] == "POSITIVE" else -sent['score']
    sentiment_tensor = torch.tensor([[sentiment_score]], dtype=torch.float32)

    with torch.no_grad():
        pred_log = model(img_tensor, input_ids, attention_mask, sentiment_tensor)
    engagement_pred = math.exp(pred_log.item()) - 1.0

    return {"engagement_score": engagement_pred}