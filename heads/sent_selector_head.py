
import torch
import torch.nn as nn

class SentenceSelectorHead(nn.Module):
    """
    Simple MLP over sentence embeddings -> 1 logit per sentence.
    Sent embeddings: mean-pooled encoder hidden states over sentence spans.
    """
    def __init__(self, d_model: int, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        h = hidden or max(128, d_model // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

    def forward(self, sent_embs: torch.Tensor) -> torch.Tensor:
        # sent_embs: (B, S, D)
        logits = self.net(sent_embs).squeeze(-1)  # (B, S)
        return logits
