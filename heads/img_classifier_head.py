# heads/img_classifier_head.py
import torch
import torch.nn as nn

class ImageClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1, pooling: str = "mean"):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden: (B,T,D), attention_mask: (B,T)
        if self.pooling == "cls":
            # take first token
            pooled = hidden[:, 0, :]
        else:
            # mean over non-pad tokens
            mask = attention_mask.float().unsqueeze(-1)  # (B,T,1)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / denom
        logits = self.fc(self.dropout(pooled))
        return logits
