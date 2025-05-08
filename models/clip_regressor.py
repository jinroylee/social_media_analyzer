import torch
import torch.nn as nn
from transformers import CLIPModel

class EngagementHead(nn.Module):
    def __init__(self, input_dim=1537, hidden_dim=1024):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class CLIPEngagementRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.head = EngagementHead()

    def forward(self, image, input_ids, attention_mask, sentiment):
        image_embeds = self.clip_model.get_image_features(pixel_values=image)
        text_embeds = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        concat = torch.cat([image_embeds, text_embeds, sentiment], dim=1)
        return self.head(concat)