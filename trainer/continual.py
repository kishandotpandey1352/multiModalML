import random
import torch
import torch.nn.functional as F
from collections import defaultdict, deque

class ReplayBuffer:
    def __init__(self, max_per_modality=256):
        self.max_per_modality = max_per_modality
        self.storage = defaultdict(lambda: deque(maxlen=max_per_modality))

    def add_batch(self, modality_name, file_path, byte_tensor):
        # store raw bytes (cpu) for later masking
        self.storage[modality_name].append((file_path, byte_tensor.cpu()))

    def sample(self, batch_size=1):
        # balanced sample across modalities
        modalities = [m for m in self.storage if len(self.storage[m]) > 0]
        if not modalities:
            return None
        picks = []
        for _ in range(batch_size):
            m = random.choice(modalities)
            fp, bt = random.choice(list(self.storage[m]))
            picks.append((m, fp, bt))
        return picks

def feature_distillation_loss(curr_features, old_features, mask=None):
    # curr_features, old_features: (B, L, D)
    if mask is not None:
        curr_features = curr_features[mask]
        old_features  = old_features[mask]
    return F.mse_loss(curr_features, old_features)

@torch.no_grad()
def compute_fisher_diag(model, data_iter, device, n_steps=200):
    # Light-weight Fisher estimate for EWC (optional)
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    steps = 0
    for batch in data_iter:
        if steps >= n_steps:
            break
        byte_input = batch['byte_input'].unsqueeze(0).to(device) if batch['byte_input'].dim()==1 else batch['byte_input'].to(device)
        mod_idx = batch['modality_index'].unsqueeze(0).to(device) if batch['modality_index'].dim()==0 else batch['modality_index'].to(device)
        out = model(byte_input, mod_idx)  # (B, L, D)
        # Use token variance as a pseudo "loglik" signal
        loss = out.var(dim=(1,2)).mean()
        loss.backward()
        for (n, p) in model.named_parameters():
            if p.grad is not None and p.requires_grad:
                fisher[n] += p.grad.detach()**2
        model.zero_grad(set_to_none=True)
        steps += 1
    for n in fisher:
        fisher[n] /= max(steps, 1)
    return fisher

def ewc_penalty(model, prev_params, fisher, lam=1000.0):
    penalty = 0.0
    for (n, p) in model.named_parameters():
        if n in prev_params and p.requires_grad:
            penalty += (fisher[n] * (p - prev_params[n]).pow(2)).sum()
    return lam * penalty
