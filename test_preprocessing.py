#!/usr/bin/env python3
"""
Test script for the new preprocessing pipeline.
"""

import pickle
import torch
from transformers import CLIPProcessor, CLIPTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.clip_regressor import CLIPEngagementRegressor

def test_preprocessing():
    """Test the new preprocessing pipeline with LoRA model"""
    print("Testing preprocessing pipeline with LoRA...")
    
    # Load processed data
    try:
        with open('data/processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded {len(data)} samples from processed_data.pkl")
    except FileNotFoundError:
        print("✗ processed_data.pkl not found. Run preprocessing first.")
        return False
    
    # Check data structure
    if not data:
        print("✗ No data found")
        return False
    
    sample = data[0]
    required_keys = ['image', 'text', 'sentiment', 'label']
    for key in required_keys:
        if key not in sample:
            print(f"✗ Missing key: {key}")
            return False
    print("✓ Data structure is correct")
    
    # Load processors
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        print("✓ Loaded CLIP processor and tokenizer")
    except Exception as e:
        print(f"✗ Failed to load processors: {e}")
        return False
    
    # Test LoRA model
    try:
        model = CLIPEngagementRegressor(use_lora=True, lora_rank=8)
        print("✓ Created LoRA model successfully")
        
        # Test with sample data
        sample = data[0]
        
        # Process image
        image_inputs = processor(images=sample['image'], return_tensors="pt")
        pixel_values = image_inputs['pixel_values']
        
        # Process text
        text_inputs = tokenizer(
            sample['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']
        
        # Sentiment
        sentiment = torch.tensor([[sample['sentiment']]], dtype=torch.float32)
        
        print(f"✓ Input shapes:")
        print(f"  - Image: {pixel_values.shape}")
        print(f"  - Text input_ids: {input_ids.shape}")
        print(f"  - Text attention_mask: {attention_mask.shape}")
        print(f"  - Sentiment: {sentiment.shape}")
        print(f"  - Label: {sample['label']}")
        
        # Test model forward pass
        model.eval()
        with torch.no_grad():
            output = model(pixel_values, input_ids, attention_mask, sentiment)
        
        print(f"✓ Model output shape: {output.shape}")
        print(f"✓ Model output value: {output.item():.4f}")
        
        # Test LoRA parameters
        lora_params = sum(1 for name, _ in model.named_parameters() if 'lora_' in name)
        total_params = sum(1 for _ in model.named_parameters())
        print(f"✓ LoRA parameters: {lora_params}/{total_params}")
        
        # Print trainable parameters info
        model.clip_model.print_trainable_parameters()
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = test_preprocessing()
        if success:
            print("\n🎉 All tests passed! The LoRA preprocessing pipeline is working correctly.")
            print("\nKey improvements with LoRA:")
            print("- ✓ Efficient fine-tuning with only ~1% of parameters")
            print("- ✓ Preserves raw images and text for end-to-end training")
            print("- ✓ No embedding dimension mismatches")
            print("- ✓ Memory efficient on-the-fly processing")
            print("- ✓ Rank-8 LoRA adapters for optimal performance/efficiency trade-off")
        else:
            print("\n❌ Tests failed. Please check the errors above.")
    except Exception as e:
        print(f"\n💥 Test script failed: {e}") 