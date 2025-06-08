import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPTokenizer
from finetuning.models.clip_regressor import CLIPEngagementRegressor
from finetuning.utils.engagement_dataset import EngagementDataset
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import pickle
import math
from PIL import Image
import sys
import os

BATCH_SIZE = 16

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    criterion = nn.HuberLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment = batch['sentiment'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(pixel_values, input_ids, attention_mask, sentiment)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            
            predictions.extend(outputs.squeeze().cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    mae = mean_absolute_error(targets, predictions)
    correlation, _ = spearmanr(targets, predictions)
    avg_loss = total_loss / len(dataloader)
    
    return mae, correlation, avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processors
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    # Load test data (separate file)
    with open("finetuning/data/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Create test dataset
    test_dataset = EngagementDataset(test_data, processor, tokenizer)
    
    print(f"Test size: {len(test_dataset)}")
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model with LoRA
    model = CLIPEngagementRegressor(use_lora=True, lora_rank=8).to(device)
    
    # Load trained model
    model_path = 'finetuning/models/best_model_lora.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please run training first to create the model.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded trained model from {model_path}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_mae, test_corr, test_loss = evaluate(model, test_loader, device)
    
    print(f'\nFinal Test Results:')
    print(f'Test MAE: {test_mae:.4f}')
    print(f'Test Correlation: {test_corr:.4f}')
    print(f'Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    main() 