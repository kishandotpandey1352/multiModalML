import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteTransformerClassifier(nn.Module):
    def __init__(self, input_len=2048, embed_dim=64, num_heads=4, num_layers=4, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)  # 256 possible byte values
        self.pos_embed = nn.Parameter(torch.randn(1, input_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(input_len * embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = (x * 255).long().clamp(0, 255)  # convert back to byte values for embedding
        x = self.embedding(x) + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.flatten(1)
        return self.classifier(x)
