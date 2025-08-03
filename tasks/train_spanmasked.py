import argparse
import glob
from loader.multiModal_dataloader import MultiModalDataset
from models.autoencoder_factory import autoencoder_factory
from decoders.span_boundary_decoder import SpanBoundaryDecoder
import torch.nn.functional as F
import os
import torch
from configurations import config
from loader.multiModal_dataloader import MODALITY_TO_INDEX
from loader.text_dataloader import ByteTextDataset
from utility.experiment_logger import log_experiment
from utility.span_masking import span_mask_input
from torch.utils.data import default_collate, DataLoader, random_split
# ------------------ ARGUMENT PARSING ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, required=True, help="One of: text, audio, image, table")
args = parser.parse_args()
config.MODALITY = args.modality

# ------------------ DEVICE SETUP ------------------
DEVICE = torch.device(config.DEVICE)

# ------------------ DATASET LOADING ------------------
train_dataset = MultiModalDataset(data_path="dataset", modality=config.MODALITY, split="train")
val_dataset = MultiModalDataset(data_path="dataset", modality=config.MODALITY, split="val")

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, collate_fn=lambda x: x)

# ------------------ MODEL SETUP ------------------
encoder = autoencoder_factory(task="encoder").to(DEVICE)
decoder = SpanBoundaryDecoder(config).to(DEVICE)

# ------------------ LOAD PREVIOUS CHECKPOINT IF EXISTS ------------------
# base_ckpt = f"checkpoints/best_spanboundary_{config.MODALITY}_{config.EMBED_DIM}d_{config.NUM_LAYERS}L"
# existing_ckpts = sorted(glob.glob(f"{base_ckpt}_V*.pt"))

# if existing_ckpts:
#     last_ckpt = existing_ckpts[-1]
#     print(f"\n✅ Loading from checkpoint: {last_ckpt}")
#     ckpt_data = torch.load(last_ckpt)
#     encoder.load_state_dict(ckpt_data['encoder'])
#     decoder.load_state_dict(ckpt_data['decoder'])
#     version = int(last_ckpt.split('_V')[-1].split('.')[0]) + 1
# else:
#     print("\n🚨 No checkpoint found, training from scratch")
#     version = 1

# new_ckpt = f"{base_ckpt}_V{version}.pt"
# config.CHECKPOINT_PATH = new_ckpt
# ------------------ LOAD FIXED VERSION CHECKPOINT ------------------
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

if os.path.exists(config.CHECKPOINT_PATH):
    print(f"\n✅ Loading from {config.CHECKPOINT_PATH}")
    ckpt_data = torch.load(config.CHECKPOINT_PATH, map_location=DEVICE)
    encoder.load_state_dict(ckpt_data['encoder'])
    decoder.load_state_dict(ckpt_data['decoder'])
else:
    print("\n🚨 No checkpoint found — starting from scratch")
# ------------------ OPTIMIZER ------------------
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config.LR)


# ------------------ TRAINING FUNCTIONS ------------------
def get_span_positions(labels):
    """Return list of (b, start, end) for each contiguous span in labels != -100"""
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

def compute_span_loss(encoded, labels, input_ids, decoder):
    span_positions = get_span_positions(labels)
    all_logits = []
    all_targets = []

    for b, start, end in span_positions:
        if start == 0 or end >= encoded.size(1) - 1:
            continue  # skip spans without valid boundaries
        left = encoded[b, start - 1]
        right = encoded[b, end + 1]
        span_len = end - start + 1
        rel_pos = torch.arange(span_len, device=encoded.device)

        left_batch = left.unsqueeze(0).repeat(span_len, 1)
        right_batch = right.unsqueeze(0).repeat(span_len, 1)

        logits = decoder(left_batch, right_batch, rel_pos)
        targets = input_ids[b, start:end + 1]
        all_logits.append(logits)
        all_targets.append(targets)

    if not all_logits:
        return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0, 0

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    loss = F.cross_entropy(logits, targets)

    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return loss, correct, total

def train_epoch():
    encoder.train()
    decoder.train()
    total_loss, total_correct, total_count = 0, 0, 0

    for batch in train_loader:
        byte_input = torch.stack([item['byte_input'] for item in batch]).to(DEVICE)
        mod_index = torch.stack([item['modality_index'] for item in batch]).to(DEVICE)

        masked, labels = span_mask_input(byte_input)
        encoded = encoder(masked, mod_index)

        loss, correct, total = compute_span_loss(encoded, labels, byte_input, decoder)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += correct
        total_count += total

    acc = 100. * total_correct / total_count
    # Diagnostics after batch loop
    print("Encoder output (mean of last batch):", encoded.mean().item())
    pred_counts = torch.bincount(byte_input[0], minlength=config.VOCAB_SIZE)
    top_preds = pred_counts.topk(5)
    print("Top target tokens:", top_preds.indices.tolist(), "Counts:", top_preds.values.tolist())
    return total_loss / len(train_loader), acc

def validate():
    encoder.eval()
    decoder.eval()
    total_loss, total_correct, total_count = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            byte_input = torch.stack([item['byte_input'] for item in batch]).to(DEVICE)
            mod_index = torch.stack([item['modality_index'] for item in batch]).to(DEVICE)

            masked, labels = span_mask_input(byte_input)
            encoded = encoder(masked, mod_index)

            loss, correct, total = compute_span_loss(encoded, labels, byte_input, decoder)

            total_loss += loss.item()
            total_correct += correct
            total_count += total

    acc = 100. * total_correct / total_count
    return total_loss / len(val_loader), acc

# ------------------ TRAIN LOOP ------------------
best_val_acc = 0.0
epochs_no_improve = 0
for epoch in range(config.EPOCHS):
        print(f"[Diagnostics] Starting Epoch {epoch + 1}")
        print(f"\n Epoch {epoch+1}/{config.EPOCHS} - Modality: {config.MODALITY}")

        train_loss, train_acc = train_epoch()
        val_loss, val_acc = validate()

        # Save if improved
        if val_acc > best_val_acc + config.EARLY_STOPPING_DELTA:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict()
                    }, config.CHECKPOINT_PATH)
            print("Checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f" No improvement for {epochs_no_improve} epoch(s)")

        # --- Early stopping condition ---
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
