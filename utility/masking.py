import torch.nn as nn
import torch
from configuration.config import MASK_PROB, MASK_TOKEN

# Masking Utility
# -----------------------------
def mask_input(input_ids, mask_prob=MASK_PROB, mask_token=MASK_TOKEN):
    labels = input_ids.clone()
    mask = torch.rand(input_ids.shape) < mask_prob
    masked_input = input_ids.clone()
    masked_input[mask] = mask_token
    labels[~mask] = -100  # Only compute loss on masked
    return masked_input, labels