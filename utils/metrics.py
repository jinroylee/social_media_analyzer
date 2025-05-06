"""
Functions for computing MAE, Spearman, and other evaluation metrics.
"""
import numpy as np
from typing import List, Dict, Union, Any
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mae(y_true: Union[List[float], np.ndarray], 
                 y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE score
    """
    return mean_absolute_error(y_true, y_pred)

def calculate_rmse(y_true: Union[List[float], np.ndarray], 
                  y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_spearman_correlation(y_true: Union[List[float], np.ndarray], 
                                  y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Spearman rank correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Spearman correlation coefficient
    """
    # Handle edge cases with all identical values
    if len(set(y_true)) <= 1 or len(set(y_pred)) <= 1:
        return 0.0
    
    correlation, _ = spearmanr(y_true, y_pred)
    return correlation

def calculate_r2(y_true: Union[List[float], np.ndarray], 
                y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Calculate R^2 (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R^2 score
    """
    return r2_score(y_true, y_pred)

def calculate_metrics(y_true: Union[List[float], np.ndarray], 
                     y_pred: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics at once.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'spearman': calculate_spearman_correlation(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred)
    }

def calculate_relative_improvement(baseline_metric: float, new_metric: float, 
                                  higher_is_better: bool = True) -> float:
    """
    Calculate relative improvement between baseline and new metric.
    
    Args:
        baseline_metric: Metric value for baseline model
        new_metric: Metric value for new model
        higher_is_better: Whether higher metric values are better
        
    Returns:
        Relative improvement as a percentage
    """
    if higher_is_better:
        # For metrics where higher is better (e.g., R^2, correlation)
        if baseline_metric == 0:
            # Avoid division by zero
            return float('inf') if new_metric > 0 else 0.0
        return ((new_metric - baseline_metric) / abs(baseline_metric)) * 100
    else:
        # For metrics where lower is better (e.g., MAE, RMSE)
        if baseline_metric == 0:
            # Avoid division by zero
            return float('-inf') if new_metric < 0 else 0.0
        return ((baseline_metric - new_metric) / abs(baseline_metric)) * 100

def calculate_error_distribution(y_true: Union[List[float], np.ndarray], 
                               y_pred: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Calculate error distribution statistics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of error distribution statistics
    """
    errors = np.array(y_pred) - np.array(y_true)
    abs_errors = np.abs(errors)
    
    return {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'mean_abs_error': np.mean(abs_errors),
        'median_abs_error': np.median(abs_errors),
        'max_abs_error': np.max(abs_errors),
        'q1_abs_error': np.percentile(abs_errors, 25),
        'q3_abs_error': np.percentile(abs_errors, 75),
    } 