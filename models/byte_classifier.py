
import torch.nn as nn

class ByteClassifier(nn.Module):
    def __init__(self, encoder, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.encoder.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.embed_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, L) ints 0..255 (255 can be MASK/PAD)
        h = self.encoder(x)   # (B, D) mean pooled
        return self.classifier(h)
