#!/bin/bash

# Install required dependencies
pip install -r requirements.txt

# Preprocess the data if not already done
python data_preprocessing/preprocess.py

# Train the PyTorch model
python training/train_torch.py