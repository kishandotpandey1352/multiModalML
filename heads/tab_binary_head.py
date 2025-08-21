# heads/tab_binary_head.py
import torch
import torch.nn as nn

def masked_mean(h: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    # h: (B,T,D), attn: (B,T)
    if attn is None:
        return h.mean(dim=1)
    m = attn.float()
    m = m / (m.sum(dim=1, keepdim=True) + 1e-9)
    return (h * m.unsqueeze(-1)).sum(dim=1)

class TabBinaryHead(nn.Module):
    """
    Simple 2-layer MLP on pooled sequence → 1 logit.
    """
    def __init__(self, hidden_dim: int, mlp_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, seq_hidden: torch.Tensor, attn: torch.Tensor | None = None) -> torch.Tensor:
        pooled = masked_mean(seq_hidden, attn)
        logit = self.proj(pooled).squeeze(-1)  # (B,)
        return logit
