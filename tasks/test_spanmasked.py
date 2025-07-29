import torch
from configurations import config
from loader.text_dataloader import ByteTextDataset
from models.autoencoder_factory import autoencoder_factory
from decoders.span_boundary_decoder import SpanBoundaryDecoder
from utility.span_masking import span_mask_input
import torch.nn.functional as F

DEVICE = config.DEVICE
CHECKPOINT_PATH = f"checkpoints/best_spanboundary_{config.MODALITY}_{config.EMBED_DIM}d_{config.NUM_LAYERS}L.pt"

# Load model
encoder = autoencoder_factory(task="encoder").to(DEVICE)
decoder = SpanBoundaryDecoder(embed_dim=config.EMBED_DIM).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval()
decoder.eval()

# Load test sample
dataset = ByteTextDataset(folder_path="dataset/text", seq_len=config.SEQ_LEN)
sample = dataset[0].unsqueeze(0).to(DEVICE)  # single sample

# Apply span masking
masked, labels = span_mask_input(sample, mask_prob=config.MASK_PROB, max_span_length=config.MASK_SPAN_LENGTH)
encoded = encoder(masked)

# Utility function
def get_span_positions(labels):
    spans = []
    for b in range(labels.size(0)):
        start = None
        for i in range(labels.size(1)):
            if labels[b, i] != -100 and start is None:
                start = i
            elif labels[b, i] == -100 and start is not None:
                spans.append((b, start, i - 1))
                start = None
        if start is not None:
            spans.append((b, start, labels.size(1) - 1))
    return spans

# Evaluate span prediction
span_positions = get_span_positions(labels)
all_logits, all_targets = [], []

for b, start, end in span_positions:
    if start == 0 or end >= encoded.size(1) - 1:
        continue  # skip spans without valid boundary
    left = encoded[b, start - 1]
    right = encoded[b, end + 1]
    span_len = end - start + 1
    rel_pos = torch.arange(span_len, device=DEVICE)
    left_batch = left.unsqueeze(0).repeat(span_len, 1)
    right_batch = right.unsqueeze(0).repeat(span_len, 1)
    logits = decoder(left_batch, right_batch, rel_pos)
    targets = sample[b, start:end + 1]
    all_logits.append(logits)
    all_targets.append(targets)

if not all_logits:
    print("⚠️ No valid spans found for evaluation.")
    exit()

logits = torch.cat(all_logits, dim=0)
targets = torch.cat(all_targets, dim=0)
preds = torch.argmax(logits, dim=-1)

correct = (preds == targets).sum().item()
total = targets.size(0)
accuracy = correct / total if total > 0 else 0

# Decode printable result
def decode_bytes(tensor):
    return ''.join([chr(b) if 32 <= b < 127 else '■' for b in tensor.tolist()])

print("\n Original (text):")
print(decode_bytes(sample[0]))
print("\n Masked input (with [■]):")
print(decode_bytes(masked[0]))
print("\n Predicted (text):")
print(decode_bytes(preds))

print(f"\n Evaluation Summary:")
print(f" Byte-level accuracy on masked spans: {accuracy:.2%}")
