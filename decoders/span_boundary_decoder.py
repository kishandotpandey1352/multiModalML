
import torch
import torch.nn as nn

# Patched decoder with logging
class SpanBoundaryDecoder(nn.Module):
    def debug_shapes(self, hidden_states):
        print(f"[Decoder] Hidden shape: {hidden_states.shape}")
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.EMBED_DIM
        self.seq_len = config.SEQ_LEN
        self.embed_pos = nn.Embedding(2 * config.SEQ_LEN, config.EMBED_DIM)

        self.sbo_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(self.embed_dim, config.VOCAB_SIZE)
        )
        
    def forward(self, left_boundary, right_boundary, relative_positions):
        pos_embed = self.embed_pos(relative_positions)
        concat = torch.cat([left_boundary, right_boundary, pos_embed], dim=-1)
        return self.sbo_layer(concat)
