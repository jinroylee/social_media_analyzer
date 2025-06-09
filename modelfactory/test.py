import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPTokenizer
from modelfactory.models.clip_regressor import CLIPEngagementRegressor
from modelfactory.utils.engagement_dataset import EngagementDataset
from modelfactory.utils.mlflow_utils import MLflowTracker
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import pickle
import math
from PIL import Image
import sys
import os
from datetime import datetime

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
    r2 = r2_score(targets, predictions)
    avg_loss = total_loss / len(dataloader)
    
    return mae, correlation, r2, avg_loss, predictions, targets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize MLflow tracker
    mlflow_tracker = MLflowTracker(
        experiment_name="social_media_engagement_prediction",
        tracking_uri=None,  # Uses local file store
        artifact_location=None
    )
    
    # Create run name with timestamp
    run_name = f"CLIP_LoRA_testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start MLflow run for testing
    with mlflow_tracker.start_run(run_name=run_name) as run:
        # Set tags for the run
        mlflow_tracker.set_tags({
            "model_type": "CLIP_LoRA",
            "task": "engagement_prediction_test",
            "framework": "pytorch",
            "device": str(device),
            "stage": "testing"
        })
        
        # Load processors
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Load test data (separate file)
        with open("modelfactory/data/test_data.pkl", "rb") as f:
            test_data = pickle.load(f)
        
        print(f"Loaded {len(test_data)} test samples")
        
        # Log test dataset size
        mlflow_tracker.log_hyperparameters({
            "test_dataset_size": len(test_data),
            "batch_size": BATCH_SIZE
        })
        
        # Create test dataset
        test_dataset = EngagementDataset(test_data, processor, tokenizer)
        
        print(f"Test size: {len(test_dataset)}")
        
        # Create test dataloader
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model with LoRA
        model = CLIPEngagementRegressor(use_lora=True, lora_rank=8).to(device)
        
        # Load trained model
        model_path = 'modelfactory/models/best_model_lora.pth'
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            print("Please run training first to create the model.")
            return
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path}")
        
        # Log model info
        mlflow_tracker.log_hyperparameters({
            "model_path": model_path,
            "model_class": model.__class__.__name__,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        })
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_mae, test_corr, test_r2, test_loss, predictions, targets = evaluate(model, test_loader, device)
        
        # Log test metrics
        test_metrics = {
            'test_mae': test_mae,
            'test_correlation': test_corr,
            'test_r2': test_r2,
            'test_loss': test_loss
        }
        
        mlflow_tracker.log_metrics(test_metrics)
        
        print(f'\nFinal Test Results:')
        print(f'Test MAE: {test_mae:.4f}')
        print(f'Test Correlation: {test_corr:.4f}')
        print(f'Test R²: {test_r2:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
        
        # Create and log test summary
        test_summary = f"""
        Test Results Summary:
        ====================
        Model: CLIP + LoRA Engagement Prediction
        Test Dataset Size: {len(test_data)}
        
        Performance Metrics:
        - Mean Absolute Error (MAE): {test_mae:.4f}
        - Spearman Correlation: {test_corr:.4f}
        - R² Score: {test_r2:.4f}
        - Test Loss (Huber): {test_loss:.4f}
        
        Device: {device}
        Model Path: {model_path}
        Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        mlflow_tracker.log_text_artifact(test_summary, "test_results_summary.txt")
        
        print(f'MLflow test run ID: {run.info.run_id}')
        print("Test results logged to MLflow!")

if __name__ == "__main__":
    main() 