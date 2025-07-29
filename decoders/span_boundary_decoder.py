import torch
import torch.nn as nn

class SpanBoundaryDecoder(nn.Module):
    """
    SpanBoundaryDecoder uses left and right span boundaries + position embedding
    to predict each token in a masked span.
    Inspired by SpanBERT's Span Boundary Objective (SBO).
    """
    def __init__(self, embed_dim, vocab_size=256, max_position=512):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2 + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, vocab_size)
        )
        self.embed_pos = nn.Embedding(max_position, embed_dim)

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
        return self.ffn(concat)  # (N, vocab_size)
