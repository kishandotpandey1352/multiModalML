import torch
import torch.nn as nn

class SpanBoundaryDecoder(nn.Module):
    """
    SpanBoundaryDecoder uses left and right span boundaries + position embedding
    to predict each token in a masked span.
    Inspired by SpanBERT's Span Boundary Objective (SBO).
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.EMBED_DIM
        self.seq_len = config.SEQ_LEN
        self.embed_pos = nn.Embedding(2 * config.SEQ_LEN, config.EMBED_DIM)

        self.sbo_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(self.embed_dim, config.VOCAB_SIZE)
        )
        
    def forward(self, left_boundary, right_boundary, relative_positions):
        """
        Arguments:
            left_boundary: (N, D) tensor – left span boundary embeddings
            right_boundary: (N, D) tensor – right span boundary embeddings
            relative_positions: (N,) – position index of each token in the span (0, 1, 2, ...)
        Returns:
            logits: (N, vocab_size)
        """
        pos_embed = self.embed_pos(relative_positions)  # (N, D)
        concat = torch.cat([left_boundary, right_boundary, pos_embed], dim=-1)  # (N, 3D)
        return self.sbo_layer(concat)  # (N, vocab_size)
