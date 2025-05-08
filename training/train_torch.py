import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import CLIPProcessor, CLIPTokenizer
from models.clip_regressor import CLIPEngagementRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import math

BATCH_SIZE = 16
EPOCHS = 10
FREEZE_EPOCHS = 5
LR = 1e-4

def evaluate(model, dataloader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for image, input_ids, attention_mask, sentiment, label in dataloader:
            pred = model(image, input_ids, attention_mask, sentiment).squeeze()
            preds.extend(pred.cpu().numpy())
            targets.extend(label.cpu().numpy())
    mae = mean_absolute_error(targets, preds)
    rho, _ = spearmanr(targets, preds)
    return mae, rho

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPEngagementRegressor().to(device)

    X, y = torch.load("data/processed_data.pt")
    N = len(X)
    val_size = int(0.15 * N)
    test_size = int(0.15 * N)
    train_size = N - val_size - test_size

    dataset = TensorDataset(X, y)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)

            image = features[:, :1024].reshape(-1, 1024)
            text = features[:, 1024:1536].reshape(-1, 512)
            sentiment = features[:, 1536:].reshape(-1, 1)

            # simulate input reshape for compatibility with model interface
            image = image.unsqueeze(1).expand(-1, 3, -1)[:, :, :224].reshape(-1, 3, 224, 224)
            input_ids = torch.zeros((image.size(0), 77), dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)

            pred = model(image.to(device), input_ids, attention_mask, sentiment.to(device)).squeeze()
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mae, val_rho = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Val MAE: {val_mae:.4f} - Spearman: {val_rho:.4f}")

    torch.save(model.state_dict(), "models/clip_engagement_model.pt")

if __name__ == "__main__":
    main()
