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

###########################
w_like = 1
w_comment = 10
w_share = 100
###########################

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

sentiment_map = {"Very Negative":0, "Negative":1, "Neutral":2, "Positive":3, "Very Positive":4}

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
    if len(comments) == 0:
        return 2  # Middle value for neutral when no comments
    
    # Join all comments into a single string for sentiment analysis
    combined_text = " ".join(comments)
    
    # Get prediction from the model
    result = sentiment_pipeline(combined_text)
    # print("comments: ", combined_text)
    # print("SA result: ", result)
    # Check if 'label' exists in the first element of result
    if 'label' in result[0]:
        label = result[0]['label']
        sentiment_class = sentiment_map[label]
        # print("true label: ", label)
        # print("sentiment class: ", sentiment_class)
    else:
        sentiment_class = 2  # Default to neutral if 'label' is not found
    
    return sentiment_class

def compute_engagement(row):
    views = max(int(row['view_count']), 1)
    like_rate = int(row['like_count']) / views
    comment_rate = int(row['comment_count']) / views
    share_rate = int(row['share_count']) / views
    reach_boost = math.log(1 + views)

    # print(like_rate, comment_rate, share_rate, reach_boost)
    score = (w_like*like_rate + w_comment*comment_rate + w_share*share_rate) * reach_boost
    # print("score: ", score)
    return score

def normalize_sentiment(sentiment_values):
    return np.array(sentiment_values) / 4.0

def normalize_engagement_labels(engagement_scores):
    scaler = MinMaxScaler()
    print("engagement_scores: ", engagement_scores[:100])
    normalized_scores = scaler.fit_transform(np.array(engagement_scores).reshape(-1, 1))
    print("normalized_scores: ", normalized_scores[:100])
    return normalized_scores.flatten()

def prepare_data(df):
    """
    Prepare data for training by saving raw inputs instead of pre-computed embeddings.
    This allows fine-tuning of the CLIP encoders.
    """
    data = []
    sentiment_scores = []
    engagement_scores = []
    skipped_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Check if image file exists
        image_path = row['thumbnail_path']
        if not os.path.exists(image_path):
            skipped_count += 1
            continue  # Skip this row if image doesn't exist
        
        # Clean text
        description = clean_text(row['description'])
        
        # Compute sentiment (discrete 0-4)
        sentiment_score = compute_sentiment(row['top_comments'])
        sentiment_scores.append(sentiment_score)
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            skipped_count += 1
            continue  # Skip this row if image can't be loaded
        
        # Compute engagement label
        engagement_score = compute_engagement(row)
        engagement_scores.append(engagement_score)
        
        # Store raw data for training (will normalize after collecting all scores)
        data_point = {
            'video_id': row['video_id'],
            'thumbnail_path': row['thumbnail_path'],
            'raw_text': row['description'],
            'top_comments': row['top_comments'],
            'view_count': row['view_count'],
            'like_count': row['like_count'],
            'comment_count': row['comment_count'],
            'share_count': row['share_count'],
            'image': image,
            'text': description,
            'sentiment': sentiment_score,
            'label': engagement_score
        }
        data.append(data_point)
    
    print(f"Skipped {skipped_count} rows due to missing or corrupted images")
    
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
    df = pd.read_parquet("data/tiktok_data/tiktok_data.parquet")

    print("Data loaded successfully")
    data = prepare_data(df)
    
    # Save processed data and scaler
    with open("data/processed_data.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print(f"Processed {len(data)} samples and saved to data/processed_data.pkl")
    
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
