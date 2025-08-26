# heads/tabqa_span_head.py
import torch
import torch.nn as nn

def mask_logits(logits: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    # attn: (B,T) where 1=keep, 0=pad
    mask = (attn == 0)
    # use a “very negative” value safe for the tensor dtype
    if logits.dtype == torch.float16:
        neg = torch.finfo(torch.float16).min / 2  # ~ -3e4
    elif logits.dtype == torch.bfloat16:
        neg = -1e4
    else:
        neg = -1e9
    return logits.masked_fill(mask, neg)

class TabQASpanHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.start_fc = nn.Linear(hidden_dim, 1)
        self.end_fc   = nn.Linear(hidden_dim, 1)

    def forward(self, seq: torch.Tensor, attn: torch.Tensor):
        # seq: [B,T,D], attn: [B,?] -> align to T
        T = seq.size(1)
        if attn.size(1) != T:
            attn = attn[:, :T]

        h = self.drop(seq)
        start = self.start_fc(h).squeeze(-1)  # [B,T]
        end   = self.end_fc(h).squeeze(-1)    # [B,T]

        start = mask_logits(start, attn)      # fp16-safe masking
        end   = mask_logits(end, attn)
        return start, end
