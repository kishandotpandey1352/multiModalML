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
        self.storage[modality_name].append((file_path, byte_tensor.detach().cpu()))

    def sample(self, batch_size: int = 1):
        mods = [m for m in self.storage if len(self.storage[m]) > 0]
        if not mods: return None
        out = []
        for _ in range(batch_size):
            m = random.choice(mods)
            out.append(random.choice(list(self.storage[m])))
        return out

def feature_distillation_loss(curr_features: torch.Tensor,
                              old_features: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(curr_features, old_features)

def compute_fisher_diag(model, data_iter, device, n_steps: int = 200):
    model.train()
    fisher = {n: torch.zeros_like(p, device=device)
              for n,p in model.named_parameters() if p.requires_grad}
    steps = 0
    for batch in data_iter:
        if steps >= n_steps: break
        x = batch["byte_input"]; m = batch["modality_index"]
        if x.dim() == 1: x = x.unsqueeze(0)
        if m.dim() == 0: m = m.unsqueeze(0)
        x = x.to(device); m = m.to(device)
        out = model(x, m)            # (B, L, D)
        loss = (out ** 2).mean()     # proxy scalar
        for p in model.parameters():
            if p.grad is not None: p.grad = None
        loss.backward()
        for (n,p) in model.named_parameters():
            if p.grad is not None and p.requires_grad:
                fisher[n] += p.grad.detach() ** 2
        steps += 1
    if steps > 0:
        for n in fisher: fisher[n] /= steps
    return fisher

def ewc_penalty(model, prev_params, fisher, lam: float = 50.0) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    if prev_params is None or fisher is None: return penalty
    for (n,p) in model.named_parameters():
        if p.requires_grad and (n in prev_params) and (n in fisher):
            prev = prev_params[n].to(p.device, non_blocking=True)
            fis  = fisher[n].to(p.device, non_blocking=True)
            penalty = penalty + (fis * (p - prev).pow(2)).sum()
    return lam * penalty
