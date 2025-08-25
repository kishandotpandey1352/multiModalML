import torch
import torch.nn as nn

class ByteEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.max_length = config["seq_len"]

        # Byte + position
        self.byte_embedding = nn.Embedding(256, self.embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.max_length, self.embed_dim))

        # Tiny modality embedding + FiLM scalers
        self.num_modalities = config.get("num_modalities", 4)
        self.modality_emb = nn.Embedding(self.num_modalities, self.embed_dim)
        self.film_gamma = nn.Linear(self.embed_dim, self.embed_dim)
        self.film_beta  = nn.Linear(self.embed_dim, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config["num_heads"],
            dim_feedforward=config["hidden_dim"],
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["num_layers"]
        )
        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(self, byte_input, modality_index):
        # byte_input: (B, L), modality_index: (B,) or scalar
        if modality_index.dim() == 0:
            modality_index = modality_index.unsqueeze(0).expand(byte_input.size(0))

        B, L = byte_input.shape
        x = self.byte_embedding(byte_input) + self.positional_embedding[:, :L, :]

        m = self.modality_emb(modality_index)           # (B, D)
        gamma = self.film_gamma(m).unsqueeze(1)         # (B, 1, D)
        beta  = self.film_beta(m).unsqueeze(1)          # (B, 1, D)
        x = (1 + gamma) * x + beta                      # FiLM

        # Return token-level features (B, L, D)
        return self.transformer_encoder(x)
