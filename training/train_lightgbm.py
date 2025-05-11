import torch
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# Load processed features
X, y = torch.load("data/processed_data.pt")
X = X.numpy()
y = y.numpy()

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM
model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1000, learning_rate=0.05)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', early_stopping_rounds=50)

# Evaluate
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
rho, _ = spearmanr(y_val, y_pred)
print(f"Baseline MAE: {mae:.4f}")
print(f"Spearman Correlation: {rho:.4f}")

# Save model
model.booster_.save_model("models/lightgbm_model.txt")