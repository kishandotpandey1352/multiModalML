# autoencoder.py

import torch
import torch.nn as nn

class ByteAutoencoder(nn.Module):
    """
    General-purpose byte-level autoencoder:
    Combines encoder + decoder for self-supervised or supervised tasks.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids):
        """
        Args:
            input_ids: (B, L) Byte token sequences
        Returns:
            Decoder output (depends on task: logits, span predictions, etc.)
        """
        z = self.encoder(input_ids)  # (B, L, D)
        return self.decoder(z)
