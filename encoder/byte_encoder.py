import torch
import torch.nn as nn

class ByteEncoder(nn.Module):
    def __init__(self, config):
        super(ByteEncoder, self).__init__()
        self.embed_dim = config["embed_dim"]
        self.max_length = config["max_length"]

        # keep vocab size at 256 to stay compatible with AG News checkpoint
        self.byte_embedding = nn.Embedding(256, self.embed_dim)

        # FIX: forward should use this same name
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.max_length, self.embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["num_layers"]
        )

        # tiny hint for callers that check this attr
        self.batch_first = True

    def forward(self, byte_input: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                return_hidden: bool = False):
        
        if attention_mask is not None and attention_mask.device != byte_input.device:
            attention_mask = attention_mask.to(byte_input.device, non_blocking=True)
        # 1) Embed
        x = self.byte_embedding(byte_input)  # (B, T, D)

        # 2) Positional encoding (use the correct attribute)
        pos = self.positional_embedding[:, :x.size(1), :]  # (1, T, D)
        x = x + pos

        # 3) Key padding mask for Transformer (True marks padding positions)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # 4) Transformer (batch_first=True)
        seq = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)  # (B, T, D)

        if return_hidden:
            return seq  # (B, T, D)

        # 5) Pooled output for classification (keeps AG News behavior)
        pooled = seq.mean(dim=1)   # (B, D)
        return pooled

    # NEW: convenience alias (doesn't change existing behavior)
    def forward_hidden(self, byte_input: torch.Tensor,
                       attention_mask: torch.Tensor | None = None):
        return self.forward(byte_input, attention_mask=attention_mask, return_hidden=True)
