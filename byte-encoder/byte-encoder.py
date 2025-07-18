import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteTransformer(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4, n_layers=2, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(256, embed_dim)  # Byte embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, byte_input):
        x = self.embed(byte_input) + self.pos_embed[:, :byte_input.size(1)]
        x = self.transformer(x)
        return x  # (batch, seq_len, embed_dim)

# --- Main code ---

batch = ["Hello world", "Multimodal 123"]
byte_data = [torch.tensor(list(bytes(s, 'utf-8'))) for s in batch]
padded = nn.utils.rnn.pad_sequence(byte_data, batch_first=True)

model = ByteTransformer()
embeddings = model(padded)  # shape: (batch, seq_len, embed_dim)

# --- Word to search ---
word = "343"
word_bytes = torch.tensor(list(bytes(word, 'utf-8'))).unsqueeze(0)  # shape: (1, word_len)
word_embedding = model(word_bytes)[0]  # shape: (word_len, embed_dim)

# Average embedding across word bytes
word_vector = word_embedding.mean(dim=0)  # shape: (embed_dim,)

# --- Similarity search ---
cos = nn.CosineSimilarity(dim=-1)
for i, sentence_embedding in enumerate(embeddings):
    similarities = cos(sentence_embedding, word_vector.unsqueeze(0))  # shape: (seq_len,)
    print("Similarity value in sentence {i}",similarities.max().item())
    if similarities.max().item() > 0.7:  # Threshold
        print(f"Word '{word}' likely found in sentence {i}: '{batch[i]}'")
    else:
        print(f"Word '{word}' not found in sentence {i}")
