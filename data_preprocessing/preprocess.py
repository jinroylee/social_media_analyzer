import os
import math
import re
import pandas as pd
import torch
from transformers import pipeline, CLIPProcessor, CLIPTokenizer
from PIL import Image
from tqdm import tqdm
import pickle

# Only load sentiment pipeline for preprocessing
sentiment_pipeline = pipeline("sentiment-analysis")

EMOJI_PATTERN = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
URL_PATTERN = re.compile(r"http\S+")

def clean_text(text):
    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = EMOJI_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_sentiment(text):
    result = sentiment_pipeline(text[:512])[0]
    score = result['score'] if result['label'] == "POSITIVE" else -result['score']
    return score

def compute_engagement(row):
    views = max(row['view_count'], 1)
    score = (
        row['like_count'] +
        row['comment_count'] +
        row['share_count'] +
        row['repost_count']
    ) / views
    return math.log(1 + score)

def prepare_data(df):
    """
    Prepare data for training by saving raw inputs instead of pre-computed embeddings.
    This allows fine-tuning of the CLIP encoders.
    """
    data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Clean text
        description = clean_text(row['description'])
        
        # Compute sentiment
        sentiment_score = compute_sentiment(description)
        
        # Load image
        image_path = row['thumbnail_path']
        image = Image.open(image_path).convert("RGB")
        
        # Compute engagement label
        label = compute_engagement(row)
        
        # Store raw data for training
        data_point = {
            'image': image,
            'text': description,
            'sentiment': sentiment_score,
            'label': label
        }
        data.append(data_point)
    
    return data

def main():
    df = pd.read_parquet("data/tiktok_data.parquet")
    data = prepare_data(df)
    
    # Save as pickle to preserve PIL Images
    with open("data/processed_data.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print(f"Processed {len(data)} samples and saved to data/processed_data.pkl")

if __name__ == "__main__":
    main()
