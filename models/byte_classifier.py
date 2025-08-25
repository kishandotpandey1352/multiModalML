import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteClassifier(nn.Module):
    def __init__(self, encoder, embed_dim=256, num_classes=4, dropout=0.1, freeze_encoder=True, pooling="mean"):
        super(ByteClassifier, self).__init__()

        self.encoder = encoder
        self.pooling = pooling
        self.embed_dim = embed_dim

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, byte_input, modality_index=None):
        """
        byte_input: Tensor of shape (batch_size, seq_len)
        modality_index: Tensor of shape (batch_size,)
        """
        hidden_states = self.encoder(byte_input, modality_index)  # (B, L, D)

        if self.pooling == "mean":
            pooled = hidden_states.mean(dim=1)
        elif self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        logits = self.classifier_head(pooled)
        return logits
