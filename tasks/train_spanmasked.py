# FINAL CLEANED train_spanmasked.py WITH LOGGING + VALIDATION SPLIT

import os
import csv
import torch
import random
from datetime import datetime
from torch import nn, optim
from loader.multiModal_dataloader import MultiModalDataset
from encoder.byte_encoder import ByteEncoder
from decoders.span_boundary_decoder import SpanBoundaryDecoder
from configurations.config import config
from torch.utils.data import DataLoader, random_split

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def span_mask_input(input_tensor, mask_prob=0.6, max_span_length=10):
    masked = input_tensor.clone()
    labels = torch.full_like(input_tensor, fill_value=-100)
    batch_size, seq_len = input_tensor.size()

    for b in range(batch_size):
        num_masked = 0
        attempts = 0
        while num_masked == 0 and attempts < 10:
            temp_masked = masked[b].clone()
            temp_labels = labels[b].clone()
            i = 0
            while i < seq_len:
                if random.random() < mask_prob:
                    span_len = random.randint(1, max_span_length)
                    span_end = min(i + span_len, seq_len)
                    temp_labels[i:span_end] = input_tensor[b, i:span_end]
                    temp_masked[i:span_end] = 0
                    i = span_end
                else:
                    i += 1
            num_masked = (temp_labels != -100).sum().item()
            attempts += 1
        masked[b] = temp_masked
        labels[b] = temp_labels
    return masked, labels

# === Model Setup ===
ckpt_path = 'checkpoints/V1/model.pth'
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
encoder = ByteEncoder(config).to(device)
decoder = SpanBoundaryDecoder(config).to(device)

if os.path.exists(ckpt_path):
    print(f"\n✅ Loading from {ckpt_path}")
    ckpt_data = torch.load(ckpt_path)
    encoder.load_state_dict(ckpt_data['encoder'])
    decoder.load_state_dict(ckpt_data['decoder'])
else:
    print("\n🚨 No checkpoint found — starting from scratch")

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# === Dataset and DataLoaders ===
full_dataset = MultiModalDataset(data_path='dataset', modality='text', split='train')
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# === Logging ===
modality = 'text'
num_samples = len(train_dataset)
epochs = config['epochs']
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"{modality}_{num_samples}samples_{epochs}epochs_{timestamp}.csv")

with open(log_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "timestamp"])

# === Training Loop ===
best_loss = float('inf')
no_improve = 0
patience = 5

for epoch in range(1, epochs + 1):
    print(f"\n📘 Epoch {epoch}/{epochs}")
    encoder.train()
    decoder.train()
    total_train_loss = 0
    train_batches = 0

    for batch in train_loader:
        byte_input = batch['byte_input'].to(device)
        modality_index = batch['modality_index'].to(device)
        path = batch['file_path'][0] if 'file_path' in batch else "(unknown)"

        masked, labels = span_mask_input(byte_input)
        masked, labels = masked.to(device), labels.to(device)

        optimizer.zero_grad()
        hidden = encoder(masked, modality_index)

        loss = 0
        count = 0
        for b in range(hidden.shape[0]):
            input_ids = byte_input[b]
            left_batch, right_batch, rel_pos, targets = [], [], [], []
            i = 0
            while i < len(labels[b]):
                if labels[b, i] != -100:
                    start = i
                    while i < len(labels[b]) and labels[b, i] != -100:
                        i += 1
                    end = i - 1
                    left = hidden[b, start - 1] if start > 0 else torch.zeros_like(hidden[b, 0])
                    right = hidden[b, end + 1] if end + 1 < hidden.size(0) else torch.zeros_like(hidden[b, 0])
                    left_batch.append(left)
                    right_batch.append(right)
                    rel_pos.append(end - start + 1)
                    targets.append(input_ids[start])
                i += 1

            if targets:
                left_batch = torch.stack(left_batch)
                right_batch = torch.stack(right_batch)
                rel_pos = torch.tensor(rel_pos, device=device)
                targets = torch.tensor(targets, device=device)
                logits = decoder(left_batch, right_batch, rel_pos)
                loss += criterion(logits, targets)
                count += 1

        if count > 0:
            loss = loss / count
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_batches += 1

    avg_train_loss = total_train_loss / max(train_batches, 1)

    # === Validation ===
    encoder.eval()
    decoder.eval()
    total_val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            byte_input = batch['byte_input'].to(device)
            modality_index = batch['modality_index'].to(device)
            masked, labels = span_mask_input(byte_input)
            masked, labels = masked.to(device), labels.to(device)
            hidden = encoder(masked, modality_index)

            loss = 0
            count = 0
            for b in range(hidden.shape[0]):
                input_ids = byte_input[b]
                left_batch, right_batch, rel_pos, targets = [], [], [], []
                i = 0
                while i < len(labels[b]):
                    if labels[b, i] != -100:
                        start = i
                        while i < len(labels[b]) and labels[b, i] != -100:
                            i += 1
                        end = i - 1
                        left = hidden[b, start - 1] if start > 0 else torch.zeros_like(hidden[b, 0])
                        right = hidden[b, end + 1] if end + 1 < hidden.size(0) else torch.zeros_like(hidden[b, 0])
                        left_batch.append(left)
                        right_batch.append(right)
                        rel_pos.append(end - start + 1)
                        targets.append(input_ids[start])
                    i += 1

                if targets:
                    left_batch = torch.stack(left_batch)
                    right_batch = torch.stack(right_batch)
                    rel_pos = torch.tensor(rel_pos, device=device)
                    targets = torch.tensor(targets, device=device)
                    logits = decoder(left_batch, right_batch, rel_pos)
                    loss += criterion(logits, targets)
                    count += 1

            if count > 0:
                loss = loss / count
                total_val_loss += loss.item()
                val_batches += 1

    avg_val_loss = total_val_loss / max(val_batches, 1)

    # === Logging to CSV ===
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_train_loss, avg_val_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    # === Early Stopping and Checkpointing ===
    if avg_val_loss < best_loss:
        print("✅ Checkpoint saved.")
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, ckpt_path)
        best_loss = avg_val_loss
        no_improve = 0
    else:
        no_improve += 1
        print(f"⚠️ No improvement for {no_improve} epoch(s)")

    if no_improve >= patience:
        print(f"⏹️ Early stopping triggered after {epoch} epochs.")
        break
