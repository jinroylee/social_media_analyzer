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
from sklearn.model_selection import train_test_split

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
    followers = max(int(row['follower_count']), 1)
    views = max(int(row['view_count']), 1)
    like_rate = int(row['like_count']) / views
    comment_rate = int(row['comment_count']) / views
    share_rate = int(row['share_count']) / views
    reach_boost = math.log(1 + views)/math.log(1 + followers)

    # print(like_rate, comment_rate, share_rate, reach_boost)
    score = (w_like*like_rate + w_comment*comment_rate + w_share*share_rate) * reach_boost
    score = views/followers
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

def split_and_save_data(data, train_ratio=0.9, random_state=42):
    """
    Split data into training and testing sets and save them separately.
    
    Args:
        data: List of processed data points
        train_ratio: Ratio for training data (default 0.9 for 9:1 split)
        random_state: Random seed for reproducibility
    """
    print(f"\nSplitting data into {train_ratio:.1%} training and {1-train_ratio:.1%} testing...")
    
    # Split the data randomly
    train_data, test_data = train_test_split(
        data, 
        train_size=train_ratio, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Create data directory if it doesn't exist
    os.makedirs("finetuning/data", exist_ok=True)
    
    # Save training data
    with open("finetuning/data/train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    print("Training data saved to finetuning/data/train_data.pkl")
    
    # Save testing data
    with open("finetuning/data/test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
    print("Testing data saved to finetuning/data/test_data.pkl")
    
    # Also save the complete dataset for backward compatibility
    with open("finetuning/data/processed_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print("Complete dataset saved to finetuning/data/processed_data.pkl")
    
    return train_data, test_data

def main():
    print("Starting data preprocessing...")
    df = pd.read_parquet("finetuning/data/tiktok_data/tiktok_data_cleaned.parquet")

    print("Data loaded successfully")
    data = prepare_data(df)
    
    # Split and save data
    train_data, test_data = split_and_save_data(data, train_ratio=0.9, random_state=42)
    
    print(f"\nProcessed {len(data)} total samples")
    
    # Print some statistics for training data
    train_sentiments = [d['sentiment'] for d in train_data]
    train_labels = [d['label'] for d in train_data]
    
    print(f"\nTraining data statistics:")
    print(f"  Sentiment - Min: {min(train_sentiments):.3f}, Max: {max(train_sentiments):.3f}")
    print(f"  Sentiment - Mean: {np.mean(train_sentiments):.3f}, Std: {np.std(train_sentiments):.3f}")
    print(f"  Engagement - Min: {min(train_labels):.3f}, Max: {max(train_labels):.3f}")
    print(f"  Engagement - Mean: {np.mean(train_labels):.3f}, Std: {np.std(train_labels):.3f}")

if __name__ == "__main__":
    main()
