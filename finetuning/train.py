import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import CLIPProcessor, CLIPTokenizer
from finetuning.models.clip_regressor import CLIPEngagementRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import pickle
import math
from PIL import Image

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
LORA_LEARNING_RATE = 1e-3  # Higher learning rate for LoRA parameters

class EngagementDataset(Dataset):
    """Dataset class for engagement prediction with raw images and text."""
    
    def __init__(self, data, processor, tokenizer):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process image
        image_inputs = self.processor(images=item['image'], return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].squeeze(0)
        
        # Process text
        text_inputs = self.tokenizer(
            item['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        
        # Sentiment
        sentiment = torch.tensor([item['sentiment']], dtype=torch.float32)
        
        # Label
        label = torch.tensor(item['label'], dtype=torch.float32)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentiment': sentiment,
            'label': label
        }

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    criterion = nn.MSELoss()
    
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
    
    # Load training data (separate file)
    with open("finetuning/data/train_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} training samples")
    
    # Create dataset
    dataset = EngagementDataset(data, processor, tokenizer)
    
    # Split training data into train/validation (80%/20% of training data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train size: {train_size}, Validation size: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model with LoRA
    model = CLIPEngagementRegressor(use_lora=True, lora_rank=8).to(device)
    
    # Setup optimizers with different learning rates for LoRA and non-LoRA parameters
    lora_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': LEARNING_RATE},
        {'params': lora_params, 'lr': LORA_LEARNING_RATE}
    ])
    
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_mae = float('inf')
    
    print("Starting training with LoRA...")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment = batch['sentiment'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values, input_ids, attention_mask, sentiment)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation
        val_mae, val_corr, val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val Correlation: {val_corr:.4f}')
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), 'finetuning/models/best_model_lora.pth')
            print(f'  New best model saved! MAE: {val_mae:.4f}')
    
    print(f'\nTraining completed!')
    print(f'Best validation MAE: {best_val_mae:.4f}')

if __name__ == "__main__":
    main()
