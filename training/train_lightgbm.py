"""
Trains and evaluates the LightGBM baseline model for social media engagement prediction.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import pickle
from sklearn.metrics import mean_absolute_error
from typing import List, Dict, Tuple, Any

from utils.metrics import calculate_spearman_correlation
from utils.config import LIGHTGBM_PARAMS

def prepare_training_data(train_df: pd.DataFrame, 
                         test_df: pd.DataFrame,
                         feature_cols: List[str], 
                         target_col: str = 'engagement') -> Tuple[Any, Any, Any, Any]:
    """
    Prepare data for LightGBM training.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        
    Returns:
        Tuple of (lgb_train, lgb_eval, y_train, y_test)
    """
    # Extract features and targets
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    return lgb_train, lgb_eval, y_train, y_test

def train_model(lgb_train: lgb.Dataset, 
               lgb_eval: lgb.Dataset, 
               params: Dict[str, Any] = None) -> lgb.Booster:
    """
    Train the LightGBM model.
    
    Args:
        lgb_train: Training dataset
        lgb_eval: Evaluation dataset
        params: LightGBM parameters
        
    Returns:
        Trained LightGBM model
    """
    if params is None:
        params = LIGHTGBM_PARAMS
    
    print(f"Training LightGBM with parameters: {params}")
    
    # Train model
    gbm = lgb.train(params,
                   lgb_train,
                   valid_sets=[lgb_train, lgb_eval],
                   callbacks=[lgb.early_stopping(stopping_rounds=100)])
    
    return gbm

def evaluate_model(model: lgb.Booster, 
                  X_test: pd.DataFrame, 
                  y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained LightGBM model
        X_test: Test features
        y_test: Test target values
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    spearman = calculate_spearman_correlation(y_test, y_pred)
    
    # Print metrics
    print(f"Model Evaluation:")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - Spearman correlation: {spearman:.4f}")
    
    return {
        'mae': mae,
        'spearman': spearman
    }

def save_model(model: lgb.Booster, model_dir: str, model_name: str = 'lightgbm_model'):
    """
    Save the trained model.
    
    Args:
        model: Trained LightGBM model
        model_dir: Directory to save the model
        model_name: Name for the saved model file
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}.txt")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(),
    })
    importance_df.sort_values(by='importance', ascending=False, inplace=True)
    importance_path = os.path.join(model_dir, f"{model_name}_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")

if __name__ == "__main__":
    # Load processed data
    train_df = pd.read_csv("../data/processed/train_data.csv")
    test_df = pd.read_csv("../data/processed/test_data.csv")
    
    # Define features and target
    feature_cols = [col for col in train_df.columns 
                   if col not in ['engagement', 'post_id', 'user_id']]
    target_col = 'engagement'
    
    # Prepare data
    lgb_train, lgb_eval, y_train, y_test = prepare_training_data(
        train_df, test_df, feature_cols, target_col
    )
    
    # Train model
    model = train_model(lgb_train, lgb_eval)
    
    # Evaluate model
    metrics = evaluate_model(model, test_df[feature_cols], y_test)
    
    # Save model
    save_model(model, "../models/saved", "lightgbm_engagement_predictor") 