# encoder.py

import torch
import torch.nn as nn

class ByteEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=256,       # Byte values: 0–255
        embed_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.byte_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) — sequence of byte tokens
        Returns:
            z: (batch_size, seq_len, embed_dim) — encoded representation
        """
        emb = self.byte_embedding(x)                    # (B, L, D)
        emb = self.positional_encoding(emb)             # (B, L, D)
        emb = emb.permute(1, 0, 2)                      # (L, B, D) for Transformer
        encoded = self.encoder(emb)                     # (L, B, D)
        return encoded.permute(1, 0, 2)                 # (B, L, D)

# -------------------------------
# Positional Encoding Module
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            x + position encoding: (batch_size, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
