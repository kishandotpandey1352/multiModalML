
import torch
import torch.nn as nn

class ByteTokenClassifierHead(nn.Module):
    """
    Per-position token classifier head: LayerNorm -> Dropout -> Linear(D -> num_tags).
    Expects encoder to return hidden states of shape (B, T, D).
    """
    def __init__(self, d_model: int, num_tags: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.cls  = nn.Linear(d_model, num_tags)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, D)
        x = self.norm(hidden_states)
        x = self.drop(x)
        logits = self.cls(x)  # (B, T, num_tags)
        return logits
