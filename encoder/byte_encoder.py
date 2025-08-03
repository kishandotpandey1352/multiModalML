
import torch
import torch.nn as nn

class ByteEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.EMBED_DIM
        self.byte_embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBED_DIM)
        self.positional_encoding = PositionalEncoding(config.EMBED_DIM, config.DROPOUT, config.SEQ_LEN)
        self.modality_embedding = nn.Embedding(config.NUM_MODALITIES, config.EMBED_DIM)

        self.ln = nn.LayerNorm(config.EMBED_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.HIDDEN_DIM,
            dropout=config.DROPOUT,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)

    def forward(self, x, modality_index):
        byte_emb = self.byte_embedding(x)              # (B, L, D)
        pos_emb = self.positional_encoding(byte_emb)   # (B, L, D)
        modality_index = modality_index.clamp(0, self.modality_embedding.num_embeddings - 1)
        mod_emb = self.modality_embedding(modality_index.long()).unsqueeze(1).expand_as(pos_emb)  # (B, L, D)
        emb = self.dropout(self.ln(pos_emb + mod_emb))  # Normalize + Dropout
        emb = emb.permute(1, 0, 2)                      # (L, B, D)
        encoded = self.encoder(emb)                     # (L, B, D)
        return encoded.permute(1, 0, 2)                 # (B, L, D)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
