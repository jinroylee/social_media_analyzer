# MLflow Experiment Tracking for Social Media Engagement Prediction

This project now includes comprehensive MLflow integration for tracking machine learning experiments, logging metrics, parameters, and model artifacts.

## 🚀 Quick Start

### 1. Install Dependencies

First, install MLflow and other dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Training with MLflow Tracking

```bash
# Run training with automatic MLflow logging
python modelfactory/train.py
```

### 3. Start MLflow UI

```bash
# Option 1: Using the management script
python scripts/mlflow_manager.py ui

# Option 2: Direct MLflow command
mlflow ui --backend-store-uri file://./mlruns
```

Access the MLflow UI at: http://localhost:5000

## 📊 What Gets Tracked

### Metrics (logged per epoch)
- **Training metrics**: loss, MAE, correlation, R²
- **Validation metrics**: loss, MAE, correlation, R²
- **Learning rates**: base and LoRA learning rates
- **Best metrics**: best validation MAE and R²

### Parameters
- **Model configuration**: LoRA rank, learning rates, batch size
- **Training setup**: epochs, optimizer, loss function
- **Dataset info**: train/validation sizes
- **Model architecture**: total and trainable parameters

### Artifacts
- **Model checkpoints**: Best model states saved per epoch
- **Final model**: Complete trained model with MLflow format
- **Training summary**: Detailed training results
- **Model architecture**: Model structure description

## 🏗️ Project Structure

```
├── modelfactory/
│   ├── utils/
│   │   └── mlflow_utils.py          # MLflow tracking utilities
│   ├── train.py                     # Training script with MLflow
│   └── test.py                      # Testing script with MLflow
├── scripts/
│   └── mlflow_manager.py            # MLflow management utilities
├── experiments/
│   └── experiment_configs.yaml      # Experiment configurations
├── mlruns/                          # MLflow tracking data (auto-created)
└── MLflow_README.md                 # This file
```

## 🧪 Managing Experiments

### Using the MLflow Manager Script

```bash
# Start MLflow UI
python scripts/mlflow_manager.py ui

# List all experiments
python scripts/mlflow_manager.py list-experiments

# List recent runs from an experiment
python scripts/mlflow_manager.py list-runs --experiment social_media_engagement_prediction

# Compare multiple runs
python scripts/mlflow_manager.py compare <run_id_1> <run_id_2> <run_id_3>

# Export the best model
python scripts/mlflow_manager.py export-best --experiment social_media_engagement_prediction

# Clean up failed runs
python scripts/mlflow_manager.py clean
```

### Experiment Organization

Experiments are automatically organized with:
- **Experiment name**: `social_media_engagement_prediction`
- **Run names**: Timestamped (e.g., `CLIP_LoRA_training_20241201_143022`)
- **Tags**: Model type, task, framework, device
- **Parameters**: All hyperparameters and configuration

## 📈 Viewing Results

### MLflow UI Features

1. **Experiments Dashboard**: Overview of all experiments
2. **Run Comparison**: Side-by-side comparison of multiple runs
3. **Metrics Plots**: Interactive plots of training progress
4. **Parameter Analysis**: Correlation between parameters and performance
5. **Artifact Browser**: Download models and other artifacts

### Key Metrics to Monitor

- **Validation MAE**: Primary metric for model selection
- **Validation R²**: Model explanation power
- **Training vs Validation**: Monitor for overfitting
- **Learning Rate Trends**: Ensure proper convergence

## 🔄 Running Different Experiments

### 1. Baseline Experiment
```bash
# Default configuration (LoRA rank=8, lr=1e-4)
python modelfactory/train.py
```

### 2. High-Rank LoRA Experiment
Modify the constants in `train.py`:
```python
LORA_RANK = 16
LORA_LEARNING_RATE = 5e-4
```

### 3. Full Fine-tuning Experiment
```python
USE_LORA = False
LEARNING_RATE = 5e-5
BATCH_SIZE = 16  # Smaller for memory
```

### 4. Different Learning Rates
```python
LEARNING_RATE = 5e-5
LORA_LEARNING_RATE = 5e-4
```

## 📋 Experiment Workflow

### 1. Planning Phase
- Define experiment objectives
- Set up experiment configuration
- Choose hyperparameters to test

### 2. Execution Phase
```bash
# Run training
python modelfactory/train.py

# Monitor in MLflow UI
python scripts/mlflow_manager.py ui
```

### 3. Analysis Phase
```bash
# List recent runs
python scripts/mlflow_manager.py list-runs

# Compare best runs
python scripts/mlflow_manager.py compare <run_id_1> <run_id_2>

# Export best model
python scripts/mlflow_manager.py export-best
```

### 4. Testing Phase
```bash
# Test the best model
python modelfactory/test.py
```

## 📊 Best Practices

### Experiment Naming
- Use descriptive run names
- Include key parameters in tags
- Group related experiments

### Hyperparameter Tracking
- Log all relevant parameters
- Use consistent parameter names
- Document parameter meanings

### Model Management
- Save checkpoints regularly
- Log model architecture details
- Include model performance summaries

### Metrics Monitoring
- Track both training and validation metrics
- Monitor learning rates and gradients
- Log custom metrics when needed

## 🔍 Troubleshooting

### Common Issues

1. **MLflow UI not starting**
   ```bash
   # Check if the mlruns directory exists
   ls -la mlruns/
   
   # Start with explicit backend store
   mlflow ui --backend-store-uri file://$(pwd)/mlruns
   ```

2. **Runs not appearing**
   - Ensure the experiment name matches
   - Check if runs completed successfully
   - Verify MLflow tracking URI

3. **Artifact upload errors**
   - Check disk space
   - Verify write permissions
   - Ensure artifact paths exist

### Performance Tips

1. **Large Models**
   - Use smaller batch sizes
   - Log checkpoints less frequently
   - Consider artifact compression

2. **Many Experiments**
   - Clean up failed runs regularly
   - Archive old experiments
   - Use external artifact storage for production

## 🎯 Advanced Usage

### Custom Metrics
```python
# In your training script
from modelfactory.utils.mlflow_utils import MLflowTracker

mlflow_tracker = MLflowTracker()
mlflow_tracker.log_metrics({
    'custom_metric': your_metric_value,
    'business_kpi': kpi_value
}, step=epoch)
```

### Model Versioning
```python
# Register model in MLflow Model Registry
mlflow.pytorch.log_model(
    model,
    "engagement_predictor",
    registered_model_name="social_media_engagement_model"
)
```

### Automated Hyperparameter Tuning
Consider integrating with:
- **Optuna**: For Bayesian optimization
- **MLflow Projects**: For reproducible runs
- **MLflow Models**: For model deployment

## 🚀 Next Steps

1. **Model Registry**: Set up MLflow Model Registry for production models
2. **Automated Training**: Create CI/CD pipelines for model training
3. **A/B Testing**: Compare models in production using MLflow
4. **Distributed Training**: Scale experiments across multiple machines

## 📚 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Best Practices for ML Experiments](https://neptune.ai/blog/ml-experiment-tracking)

---

For questions or issues with the MLflow integration, please check the troubleshooting section above or refer to the MLflow documentation. 