# trainer/continual.py
import random
from collections import defaultdict, deque
import torch
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, max_per_modality: int = 256):
        self.max_per_modality = max_per_modality
        self.storage = defaultdict(lambda: deque(maxlen=max_per_modality))

    def add_batch(self, modality_name: str, file_path: str, byte_tensor: torch.Tensor):
        # store CPU copy
        self.storage[modality_name].append((file_path, byte_tensor.detach().cpu()))

    def sample(self, batch_size: int = 1):
        modalities = [m for m in self.storage if len(self.storage[m]) > 0]
        if not modalities:
            return None
        picks = []
        for _ in range(batch_size):
            m = random.choice(modalities)
            fp, bt = random.choice(list(self.storage[m]))
            picks.append((m, fp, bt))
        return picks

def feature_distillation_loss(curr_features: torch.Tensor,
                              old_features: torch.Tensor) -> torch.Tensor:
    """
    MSE over token features (B, L, D). Shapes must match.
    """
    return F.mse_loss(curr_features, old_features)

def compute_fisher_diag(model, data_iter, device, n_steps: int = 200):
    """
    Estimate diagonal Fisher using a simple proxy scalar loss that flows through
    the encoder outputs. Requires grads ON (no torch.no_grad).
    Expects each batch to be a dict with keys: 'byte_input', 'modality_index'.
    """
    model.train()
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if p.requires_grad}
    steps = 0

    for batch in data_iter:
        if steps >= n_steps:
            break

        b = batch
        byte_input = b["byte_input"]
        mod_idx    = b["modality_index"]
        # normalize to batch first
        if byte_input.dim() == 1:
            byte_input = byte_input.unsqueeze(0)
        if mod_idx.dim() == 0:
            mod_idx = mod_idx.unsqueeze(0)

        byte_input = byte_input.to(device)
        mod_idx    = mod_idx.to(device)

        out = model(byte_input, mod_idx)  # (B, L, D)
        # simple proxy scalar loss that touches the whole graph
        loss = (out ** 2).mean()

        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        loss.backward()

        for (n, p) in model.named_parameters():
            if p.grad is not None and p.requires_grad:
                fisher[n] += p.grad.detach() ** 2

        steps += 1

    if steps > 0:
        for n in fisher:
            fisher[n] /= steps

    return fisher

def ewc_penalty(model, prev_params, fisher, lam: float = 1000.0) -> torch.Tensor:
    """
    Device-safe EWC penalty: moves snapshots to p.device on the fly.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    if prev_params is None or fisher is None:
        return penalty
    for (n, p) in model.named_parameters():
        if p.requires_grad and (n in prev_params) and (n in fisher):
            prev = prev_params[n].to(p.device, non_blocking=True)
            fis  = fisher[n].to(p.device, non_blocking=True)
            penalty = penalty + (fis * (p - prev).pow(2)).sum()
    return lam * penalty
