import os
import torch
from torch.utils.data import DataLoader, random_split
from configurations import config
from loader.text_dataloader import ByteTextDataset
from models.autoencoder_factory import autoencoder_factory
from decoders.span_boundary_decoder import SpanBoundaryDecoder
from utility.experiment_logger import log_experiment
from utility.span_masking import span_mask_input
import torch.nn.functional as F

DEVICE = config.DEVICE
VAL_SPLIT = 0.1
TOTAL_EPOCHS = config.EPOCHS
PATIENCE = 10
CHECKPOINT_PATH = f"checkpoints/best_spanboundary_{config.MODALITY}_{config.EMBED_DIM}d_{config.NUM_LAYERS}L.pt"

dataset = ByteTextDataset(folder_path="dataset/text", seq_len=config.SEQ_LEN)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

encoder = autoencoder_factory(task="encoder").to(DEVICE)
decoder = SpanBoundaryDecoder(embed_dim=config.EMBED_DIM).to(DEVICE)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=config.LR)


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


def compute_span_loss(encoded, labels, input_ids):
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
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        masked, labels = span_mask_input(batch, mask_prob=config.MASK_PROB, max_span_length=config.MASK_SPAN_LENGTH)
        encoded = encoder(masked)
        loss, correct, total = compute_span_loss(encoded, labels, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += correct
        total_tokens += total

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return total_loss / len(train_loader), accuracy


def validate():
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            masked, labels = span_mask_input(batch, mask_prob=config.MASK_PROB, max_span_length=config.MASK_SPAN_LENGTH)
            encoded = encoder(masked)
            loss, correct, total = compute_span_loss(encoded, labels, batch)

            total_loss += loss.item()
            total_correct += correct
            total_tokens += total

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return total_loss / len(val_loader), accuracy


best_accuracy = 0
patience_counter = 0

for epoch in range(1, TOTAL_EPOCHS + 1):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = validate()

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        patience_counter = 0
        torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, CHECKPOINT_PATH)
        print("✅ Best model updated.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

    log_experiment(
        dataset_name=f"{config.MODALITY}-{len(dataset)}",
        num_files=len(dataset),
        mask_prob=config.MASK_PROB,
        mask_strategy="span-boundary",
        embed_dim=config.EMBED_DIM,
        num_layers=config.NUM_LAYERS,
        hidden_dim=config.HIDDEN_DIM,
        num_heads=config.NUM_HEADS,
        epochs=epoch,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LR,
        accuracy=val_acc,
        loss=val_loss,
        notes="SpanBERT SBO training"
    )
