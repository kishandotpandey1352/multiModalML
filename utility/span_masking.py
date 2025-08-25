# utility/span_masking.py
import torch
import random

def _normalize_to_BL(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize input to shape (B, L).
    Accepts 1D (L,), 2D (B,L), or >=3D where last dim is L.
    """
    if x.dim() == 1:
        return x.unsqueeze(0)  # (1, L)
    if x.dim() == 2:
        return x               # (B, L)
    # flatten all leading dims into batch; keep last as seq
    return x.view(-1, x.size(-1))

def span_mask_input(input_tensor, mask_prob=0.6, max_span_length=10, ignore_index: int = -100):
    """
    Span-mask bytes. Returns (masked, labels) each shaped (B, L).
    - masked: original with masked spans set to 0
    - labels: original values for masked positions, ignore_index elsewhere
    """
    x = input_tensor
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.long)

    x = _normalize_to_BL(x).long()
    masked = x.clone()
    labels = torch.full_like(x, fill_value=ignore_index)

    B, L = x.size()
    for b in range(B):
        num_masked, attempts = 0, 0
        # ensure at least one masked token so the task isn't degenerate
        while num_masked == 0 and attempts < 10:
            tmp_m = masked[b].clone()
            tmp_y = labels[b].clone()
            i = 0
            while i < L:
                if random.random() < mask_prob:
                    span_len = random.randint(1, max_span_length)
                    j = min(i + span_len, L)
                    tmp_y[i:j] = x[b, i:j]
                    tmp_m[i:j] = 0
                    i = j
                else:
                    i += 1
            num_masked = (tmp_y != ignore_index).sum().item()
            attempts += 1
        masked[b] = tmp_m
        labels[b] = tmp_y
    return masked, labels
