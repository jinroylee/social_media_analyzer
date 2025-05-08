import os
import math
import re
import pandas as pd
import torch
from transformers import pipeline, CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
from tqdm import tqdm

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
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

def extract_features(df):
    features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        description = clean_text(row['description'])
        sentiment_score = compute_sentiment(description)

        text_inputs = clip_tokenizer(description, truncation=True, max_length=77, return_tensors="pt")
        text_embeds = clip_model.get_text_features(**text_inputs).squeeze().detach()

        image_path = row['thumbnail_path']
        image = Image.open(image_path).convert("RGB")
        image_inputs = clip_processor(images=image, return_tensors="pt")
        image_embeds = clip_model.get_image_features(**image_inputs).squeeze().detach()

        feat = torch.cat([image_embeds, text_embeds, torch.tensor([sentiment_score])])
        features.append(feat)

        label = compute_engagement(row)
        labels.append(label)

    return torch.stack(features), torch.tensor(labels)

def main():
    df = pd.read_parquet("data/tiktok_data.parquet")
    features, labels = extract_features(df)
    torch.save((features, labels), "data/processed_data.pt")

if __name__ == "__main__":
    main()
