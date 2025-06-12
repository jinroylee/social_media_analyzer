# Architecture Fix: Enabling CLIP Fine-tuning

## Problem Identified

You correctly identified a critical architectural mismatch in the codebase:

### Before (Problematic Architecture):

1. **Preprocessing** (`preprocess.py`):
   - Loaded CLIP model and computed embeddings
   - Saved pre-computed embeddings (1024-dim image + 512-dim text + 1-dim sentiment = 1537 total)
   - Lost access to raw images and text

2. **Model** (`clip_regressor.py`):
   - Expected raw images and text tokens as input
   - Had its own CLIP model for encoding
   - Input dimension mismatch with preprocessing

3. **Training** (`train_torch.py`):
   - Loaded pre-computed embeddings
   - Tried to fake raw inputs by reshaping embeddings back to images/tokens
   - **This prevented fine-tuning of CLIP encoders!**

### The Core Issue:
- Preprocessing computed embeddings with one CLIP model
- Training used a different CLIP model instance
- No gradient flow to CLIP encoders = no fine-tuning possible

## Solution Implemented

### New Architecture:

1. **Preprocessing** (`preprocess.py`):
   ```python
   # Now saves raw data instead of embeddings
   data_point = {
       'image': image,           # PIL Image object
       'text': description,      # Cleaned text string
       'sentiment': sentiment_score,  # Float
       'label': label           # Float
   }
   ```

2. **Training** (`train_torch.py`):
   ```python
   # Custom Dataset class that processes raw data on-the-fly
   class EngagementDataset(Dataset):
       def __getitem__(self, idx):
           # Process image with CLIP processor
           image_inputs = self.processor(images=item['image'], return_tensors="pt")
           # Process text with CLIP tokenizer
           text_inputs = self.tokenizer(item['text'], ...)
           return pixel_values, input_ids, attention_mask, sentiment, label
   ```

3. **Model** (`clip_regressor.py`):
   - Unchanged - already expected raw inputs
   - Now receives proper raw data for encoding

## Key Improvements

### 1. **Fine-tuning Enabled**
```python
# Freeze CLIP initially, then unfreeze for fine-tuning
for param in model.clip_model.parameters():
    param.requires_grad = False  # Initial training

# Later...
for param in model.clip_model.parameters():
    param.requires_grad = True   # Fine-tuning enabled!
```

### 2. **Proper Training Strategy**
- **Epochs 1-5**: Train only the engagement head (CLIP frozen)
- **Epochs 6-10**: Fine-tune entire model including CLIP encoders
- Lower learning rate for fine-tuning phase

### 3. **Consistent Data Flow**
```
Raw Image → CLIP Vision Encoder → Image Embeddings ↘
                                                    → Concat → Engagement Head → Score
Raw Text  → CLIP Text Encoder  → Text Embeddings  ↗
```

### 4. **Memory Efficiency**
- Process data on-the-fly instead of storing large embedding tensors
- Use DataLoader with num_workers for parallel processing

## Files Modified

1. **`data_preprocessing/preprocess.py`**:
   - Removed CLIP model loading
   - Save raw PIL images and cleaned text
   - Use pickle format to preserve image objects

2. **`training/train_torch.py`**:
   - Complete rewrite with custom Dataset class
   - Proper fine-tuning strategy
   - Better evaluation and model saving

3. **`serving/app.py`**:
   - Added text cleaning to match preprocessing
   - Removed redundant CLIP model loading

4. **`test_preprocessing.py`** (new):
   - Validation script to ensure pipeline works correctly

## Benefits Achieved

✅ **CLIP encoders can now be fine-tuned end-to-end**
✅ **No more embedding dimension mismatches**
✅ **Consistent text processing between training and inference**
✅ **Better training strategy with gradual unfreezing**
✅ **More memory efficient data loading**
✅ **Proper gradient flow throughout the model**

## Usage

1. **Reprocess your data**:
   ```bash
   python data_preprocessing/preprocess.py
   ```

2. **Train with fine-tuning**:
   ```bash
   python training/train_torch.py
   ```

3. **Test the pipeline**:
   ```bash
   python test_preprocessing.py
   ```

The model will now properly fine-tune the CLIP encoders for your specific engagement prediction task, which should lead to significantly better performance! 