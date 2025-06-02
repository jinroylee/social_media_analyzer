import os
import math
import re
import pandas as pd
import torch
from transformers import pipeline
from PIL import Image
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Only load sentiment pipeline for preprocessing
sentiment_pipeline = pipeline(
    "text-classification", 
    model="tabularisai/multilingual-sentiment-analysis",
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    batch_size=32,  # Process multiple texts at once
    return_all_scores=False
)

EMOJI_PATTERN = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
URL_PATTERN = re.compile(r"http\S+")

def clean_text(text):
    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = EMOJI_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_sentiment(comments):
    """
    Compute sentiment score as discrete value from 0-4.
    If no comments exist, return middle value (2).
    """
    if not comments or len(comments) == 0:
        return 2  # Middle value for neutral when no comments
    
    # Get prediction from the model
    result = sentiment_pipeline(comments.join(" "))
    print("SA result: ", result)
    # Extract the predicted class (0-4)
    # The model returns labels like "LABEL_0", "LABEL_1", etc.
    if 'label' in result:
        label = result['label']
        if label.startswith('LABEL_'):
            sentiment_class = int(label.split('_')[1])
        else:
            # Handle different label formats
            label_map = {
                "Very Negative": 0,
                "Negative": 1, 
                "Neutral": 2,
                "Positive": 3,
                "Very Positive": 4
            }
            sentiment_class = label_map.get(label, 2)  # Default to neutral
    else:
        sentiment_class = 2  # Default to neutral if no label found
    
    return sentiment_class

def compute_engagement(row):
    views = max(row['view_count'], 1)
    like_rate = row['like_count'] / views
    comment_rate = row['comment_count'] / views
    share_rate = row['share_count'] / views
    reach_boost = math.log(1 + views)

    score = (like_rate + comment_rate + share_rate) * reach_boost
    return score

def normalize_sentiment(sentiment_values):
    return np.array(sentiment_values) / 4.0

def normalize_engagement_labels(engagement_scores):
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(np.array(engagement_scores).reshape(-1, 1))
    return normalized_scores.flatten()

def prepare_data(df):
    """
    Prepare data for training by saving raw inputs instead of pre-computed embeddings.
    This allows fine-tuning of the CLIP encoders.
    """
    data = []
    sentiment_scores = []
    engagement_scores = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Clean text
        description = clean_text(row['description'])
        
        # Compute sentiment (discrete 0-4)
        sentiment_score = compute_sentiment(row['top_comments'])
        sentiment_scores.append(sentiment_score)
        
        # Load image
        image_path = row['thumbnail_path']
        image = Image.open(image_path).convert("RGB")
        
        # Compute engagement label
        engagement_score = compute_engagement(row)
        engagement_scores.append(engagement_score)
        
        # Store raw data for training (will normalize after collecting all scores)
        data_point = {
            'image': image,
            'text': description,
            'sentiment': sentiment_score,
            'label': engagement_score
        }
        data.append(data_point)
    
    # Normalize sentiment scores (0-4 -> 0-1 with 2->0.5)
    normalized_sentiments = normalize_sentiment(sentiment_scores)
    
    # MinMax normalize engagement labels
    normalized_engagement = normalize_engagement_labels(engagement_scores)
    
    # Update data with normalized values
    for i, data_point in enumerate(data):
        data_point['sentiment'] = normalized_sentiments[i]
        data_point['label'] = normalized_engagement[i]
    
    return data

def main():
    print("Starting data preprocessing...")
    df = pd.read_parquet("data/tiktok_data.parquet")

    print("Data loaded successfully")
    data = prepare_data(df)
    
    # Save processed data and scaler
    with open("data/processed_data.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print(f"Processed {len(data)} samples and saved to data/processed_data.pkl")
    print("Engagement scaler saved to data/engagement_scaler.pkl")
    
    # Print some statistics
    sentiments = [d['sentiment'] for d in data]
    labels = [d['label'] for d in data]
    
    print(f"\nSentiment statistics:")
    print(f"  Min: {min(sentiments):.3f}, Max: {max(sentiments):.3f}")
    print(f"  Mean: {np.mean(sentiments):.3f}, Std: {np.std(sentiments):.3f}")
    
    print(f"\nEngagement label statistics:")
    print(f"  Min: {min(labels):.3f}, Max: {max(labels):.3f}")
    print(f"  Mean: {np.mean(labels):.3f}, Std: {np.std(labels):.3f}")

if __name__ == "__main__":
    main()
