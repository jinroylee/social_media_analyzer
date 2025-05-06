"""
Evaluation script for the PyTorch CLIP-based engagement prediction model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

from models.clip_regressor import CLIPEngagementRegressor
from training.train_torch import SocialMediaDataset
from utils.metrics import calculate_spearman_correlation
from utils.config import TORCH_TRAINING_PARAMS

def load_model(model_path: str, model_config: Dict[str, Any] = None) -> CLIPEngagementRegressor:
    """
    Load a trained model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        model_config: Configuration for initializing the model
        
    Returns:
        Loaded model
    """
    if model_config is None:
        model_config = {
            'clip_model_name': 'openai/clip-vit-base-patch32',
            'freeze_clip': True
        }
    
    # Initialize model
    model = CLIPEngagementRegressor(
        clip_model_name=model_config['clip_model_name'],
        freeze_clip=model_config['freeze_clip']
    )
    
    # Load state dict
    model.load_state_dict(torch.load(model_path))
    
    return model

def evaluate_model(model: nn.Module, 
                  dataloader: DataLoader,
                  device: torch.device = None) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Store predictions and targets
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    spearman = calculate_spearman_correlation(all_targets, all_preds)
    mse = np.mean(np.square(np.array(all_preds) - np.array(all_targets)))
    rmse = np.sqrt(mse)
    
    metrics = {
        'mae': float(mae),
        'spearman': float(spearman),
        'mse': float(mse),
        'rmse': float(rmse)
    }
    
    return metrics, all_targets, all_preds

def plot_predictions(targets: List[float], 
                    predictions: List[float],
                    save_path: str = 'prediction_plot.png'):
    """
    Plot the predictions against the ground truth.
    
    Args:
        targets: List of ground truth values
        predictions: List of predictions
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Engagement')
    plt.ylabel('Predicted Engagement')
    plt.title('True vs Predicted Engagement')
    
    # Add correlation coefficient
    spearman = calculate_spearman_correlation(targets, predictions)
    plt.text(0.05, 0.95, f'Spearman Correlation: {spearman:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Plot saved to {save_path}")

def analyze_errors(df: pd.DataFrame, 
                  targets: List[float], 
                  predictions: List[float],
                  text_col: str = 'caption',
                  image_col: str = 'image_path',
                  n_worst: int = 10) -> pd.DataFrame:
    """
    Analyze the worst errors to understand where the model is failing.
    
    Args:
        df: DataFrame containing the test data
        targets: List of ground truth values
        predictions: List of predictions
        text_col: Column containing text/captions
        image_col: Column containing image paths
        n_worst: Number of worst predictions to analyze
        
    Returns:
        DataFrame with the worst predictions and errors
    """
    # Calculate absolute errors
    errors = np.abs(np.array(predictions) - np.array(targets))
    
    # Create DataFrame with errors
    error_df = df.copy()
    error_df['true_engagement'] = targets
    error_df['predicted_engagement'] = predictions
    error_df['absolute_error'] = errors
    
    # Sort by error (descending)
    error_df = error_df.sort_values('absolute_error', ascending=False)
    
    # Return top N worst predictions
    return error_df.head(n_worst)

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    params = TORCH_TRAINING_PARAMS
    
    # Load model
    model_path = os.path.join(params['save_dir'], 'best_clip_regressor.pt')
    model = load_model(model_path, model_config={
        'clip_model_name': params['clip_model_name'],
        'freeze_clip': params['freeze_clip']
    })
    
    # Load test data
    test_df = pd.read_csv("../data/processed/test_data.csv")
    
    # Create dataset and dataloader
    test_dataset = SocialMediaDataset(
        test_df, 
        image_dir=params['image_dir'],
        processor=model.processor,
        image_col=params['image_col'],
        text_col=params['text_col'],
        target_col=params['target_col']
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['num_workers']
    )
    
    # Evaluate model
    metrics, targets, predictions = evaluate_model(model, test_dataloader, device)
    
    # Print metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value:.4f}")
    
    # Create results directory
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(results_dir, 'torch_model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot predictions
    plot_predictions(targets, predictions, os.path.join(results_dir, 'predictions_plot.png'))
    
    # Analyze errors
    error_analysis = analyze_errors(test_df, targets, predictions)
    error_analysis.to_csv(os.path.join(results_dir, 'error_analysis.csv'), index=False)
    
    print("Evaluation complete!") 