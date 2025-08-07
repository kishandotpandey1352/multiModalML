import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteClassifier(nn.Module):
    def __init__(self, encoder, embed_dim=256, num_classes=4, dropout=0.1):
        super(ByteClassifier, self).__init__()

        # Freeze the encoder
        for param in encoder.parameters():
            param.requires_grad = False

        self.encoder = encoder
        self.embed_dim = embed_dim
        self.dropout = dropout

        # Classification head
        self.pooling = "mean"  # or "cls" if using special tokens
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, byte_input, modality_index=None):
        """
        byte_input: Tensor of shape (batch_size, seq_len)
        modality_index: Optional tensor for encoder (shape: [batch_size])
        """
        # Get hidden states from the frozen encoder
        with torch.no_grad():
            hidden_states = self.encoder(byte_input, modality_index)  # (B, L, D)

        if self.pooling == "mean":
            pooled = hidden_states.mean(dim=1)  # (B, D)
        elif self.pooling == "cls":
            pooled = hidden_states[:, 0]  # CLS token (if used)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        logits = self.classifier(pooled)
        return logits
