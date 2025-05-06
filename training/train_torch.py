"""
Trains the PyTorch end-to-end CLIP-based model for social media engagement prediction.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from typing import Dict, List, Tuple, Union, Optional

from models.clip_regressor import CLIPEngagementRegressor
from utils.metrics import calculate_spearman_correlation
from utils.config import TORCH_TRAINING_PARAMS

class SocialMediaDataset(Dataset):
    """
    Dataset for social media posts with images and captions.
    """
    def __init__(self, 
                df: pd.DataFrame, 
                image_dir: str,
                processor=None, 
                image_col: str = 'image_path',
                text_col: str = 'caption',
                target_col: str = 'engagement'):
        """
        Initialize the dataset.
        
        Args:
            df: DataFrame containing the data
            image_dir: Directory containing the images
            processor: CLIP processor for preprocessing
            image_col: Column containing image filenames
            text_col: Column containing text/captions
            target_col: Column containing target values
        """
        self.df = df
        self.image_dir = image_dir
        self.processor = processor
        self.image_col = image_col
        self.text_col = text_col
        self.target_col = target_col
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image and text
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row[self.image_col])
        text = row[self.text_col]
        target = row[self.target_col]
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # If processor is provided, process inputs
        if self.processor:
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            
            # Remove batch dimension
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs['target'] = torch.tensor(target, dtype=torch.float32)
            
            return inputs
        
        # Otherwise return raw data
        return {
            'image': image,
            'text': text,
            'target': torch.tensor(target, dtype=torch.float32)
        }

def train_epoch(model: nn.Module, 
               dataloader: DataLoader, 
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               device: torch.device) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        # Move inputs to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        targets = batch['target'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model: nn.Module, 
            dataloader: DataLoader, 
            criterion: nn.Module,
            device: torch.device) -> Tuple[float, float, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average validation loss, MAE, Spearman correlation)
    """
    model.eval()
    total_loss = 0.0
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
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            
            # Store predictions and targets
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    spearman = calculate_spearman_correlation(all_targets, all_preds)
    
    return avg_loss, mae, spearman

def train_model(model: nn.Module, 
               train_dataloader: DataLoader,
               val_dataloader: DataLoader,
               num_epochs: int = 10,
               lr: float = 1e-4,
               device: torch.device = None,
               save_dir: str = 'saved_models') -> nn.Module:
    """
    Train the model.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        num_epochs: Number of epochs to train for
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save model checkpoints
        
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize tracking variables
    best_val_spearman = -float('inf')
    
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_mae, val_spearman = validate(model, val_dataloader, criterion, device)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val Spearman: {val_spearman:.4f}")
        
        # Save best model
        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_clip_regressor.pt'))
            print(f"  Saved new best model with Spearman: {val_spearman:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_clip_regressor.pt')))
    
    return model

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    params = TORCH_TRAINING_PARAMS
    
    # Initialize model
    model = CLIPEngagementRegressor(
        clip_model_name=params['clip_model_name'],
        freeze_clip=params['freeze_clip']
    )
    
    # Load data
    train_df = pd.read_csv("../data/processed/train_data.csv")
    val_df = pd.read_csv("../data/processed/val_data.csv")
    
    # Create datasets
    train_dataset = SocialMediaDataset(
        train_df, 
        image_dir=params['image_dir'],
        processor=model.processor,
        image_col=params['image_col'],
        text_col=params['text_col'],
        target_col=params['target_col']
    )
    
    val_dataset = SocialMediaDataset(
        val_df, 
        image_dir=params['image_dir'],
        processor=model.processor,
        image_col=params['image_col'],
        text_col=params['text_col'],
        target_col=params['target_col']
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=params['num_workers']
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['num_workers']
    )
    
    # Train model
    model = train_model(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs=params['num_epochs'],
        lr=params['learning_rate'],
        device=device,
        save_dir=params['save_dir']
    )
    
    print("Training complete!") 