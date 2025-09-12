# heads/tab_binary_head.py
import torch
import torch.nn as nn

class TabBinaryHead(nn.Module):
    def __init__(self, hidden_dim: int, hidden: int = 256, p: float = 0.1):
        super().__init__()
        self.in_features = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # Expect [B, F] with F == self.in_features
        if pooled.dim() == 1:
            # If someone passed [F], treat as single batch
            pooled = pooled.unsqueeze(0)

        if pooled.dim() != 2:
            raise ValueError(f"[TabBinaryHead] expected 2D [B,F], got {tuple(pooled.shape)}")

        B, F = pooled.shape
        if F != self.in_features:
            raise ValueError(
                f"[TabBinaryHead] in_features mismatch: expected {self.in_features}, "
                f"got {F}. Did you accidentally reduce or squeeze the feature dim?"
            )

        # Match dtype (fp32/fp16) to layer weights
        pooled = pooled.to(dtype=self.proj[0].weight.dtype)
        logit = self.proj(pooled).squeeze(-1)  # [B]
        return logit
