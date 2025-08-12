
import torch
import torch.nn as nn

class ByteEncoder(nn.Module):
    def __init__(self, config):
        super(ByteEncoder, self).__init__()
        self.embed_dim = config["embed_dim"]
        self.max_length = config["max_length"]
        self.byte_embedding = nn.Embedding(256, self.embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.max_length, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])

    def forward(self, byte_input):
        # byte_input: (B, L) ints 0..255 (255 may be PAD/MASK)
        B, L = byte_input.shape
        x = self.byte_embedding(byte_input)  # (B, L, D)
        x = x + self.positional_embedding[:, :L, :]
        x = self.transformer_encoder(x)      # (B, L, D)
        return x.mean(dim=1)                 # (B, D), mean pool like pretraining
