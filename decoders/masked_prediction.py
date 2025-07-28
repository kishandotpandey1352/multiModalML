# masked_prediction.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedPredictionDecoder(nn.Module):
    def __init__(self, embed_dim=128, vocab_size=256):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoded):
        """
        Args:
            encoded: (B, L, D) from encoder
        Returns:
            logits: (B, L, vocab_size)
        """
        return self.classifier(encoded)


def masked_byte_loss(logits, labels, ignore_index=-100):
    """
    Computes cross-entropy loss over masked positions only.

    Args:
        logits: (B, L, vocab_size) – output from decoder
        labels: (B, L) – original byte values with masked positions labeled; unmasked = -100
        ignore_index: value to ignore in loss (default: -100)

    Returns:
        loss: scalar
    """
    B, L, V = logits.shape
    return F.cross_entropy(logits.view(B * L, V), labels.view(B * L), ignore_index=ignore_index)
