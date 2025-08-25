# train_spanmasked.py — simple trainer (unified masking + argparse + boundary fix)

import os
import csv
import torch
from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import argparse

from configurations.config import config as CFG
from loader.multiModal_dataloader import MultiModalDataset
from encoder.byte_encoder import ByteEncoder

# decoder import: prefer package path, fallback to local file if needed
try:
    from decoders.span_boundary_decoder import SpanBoundaryDecoder
except ModuleNotFoundError:
    from decoders.span_boundary_decoder import SpanBoundaryDecoder

# IMPORTANT: use the unified masker (supports ignore_index)
from utility.span_masking import span_mask_input

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default=CFG["modality"])
    parser.add_argument('--data_path', type=str, default=CFG["data_path"])
    parser.add_argument('--checkpoint', type=str, default=CFG["checkpoint_path"])
    args = parser.parse_args()

    device = torch.device(CFG.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # === Model ===
    ckpt_path = args.checkpoint
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    encoder = ByteEncoder(CFG).to(device)
    decoder = SpanBoundaryDecoder(CFG).to(device)

    if os.path.exists(ckpt_path):
        print(f"\n✅ Loading from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
    else:
        print("\n🚨 No checkpoint found — starting from scratch")

    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=CFG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=CFG["ignore_index"])

    # === Data ===
    full_dataset = MultiModalDataset(data_path=args.data_path, modality=args.modality, split='train')
    train_size = int(0.9 * len(full_dataset)) if len(full_dataset) > 1 else len(full_dataset)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"], shuffle=False)

    # === Logging ===
    modality = args.modality
    num_samples = len(train_dataset)
    epochs = CFG["epochs"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"{modality}_{num_samples}samples_{epochs}epochs_{timestamp}.csv")
    with open(log_path, mode='w', newline='') as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "timestamp"])

    # === Train ===
    best_loss = float('inf')
    no_improve = 0
    patience = CFG["early_stopping_patience"]

    for epoch in range(1, epochs + 1):
        print(f"\n📘 Epoch {epoch}/{epochs}")
        encoder.train(); decoder.train()
        total_train_loss, train_batches = 0.0, 0

        for batch in train_loader:
            byte_input = batch['byte_input'].to(device)           # (B,L)
            mod_idx    = batch['modality_index'].to(device)       # (B,)

            # unified span mask — supports ignore_index/mask_prob/span_length
            masked, labels = span_mask_input(
                byte_input,
                mask_prob=CFG["mask_prob"],
                max_span_length=CFG["mask_span_length"],
                ignore_index=CFG["ignore_index"]
            )
            masked, labels = masked.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            hidden = encoder(masked, mod_idx)  # (B,L,D)

            # supervised span-boundary reconstruction
            loss = torch.tensor(0.0, device=device)
            count = 0
            B, L, D = hidden.shape
            for b in range(B):
                inp = byte_input[b]
                lbl = labels[b]
                i = 0
                left_batch, right_batch, rel_pos, targets = [], [], [], []
                while i < L:
                    if lbl[i] != CFG["ignore_index"]:
                        start = i
                        while i < L and lbl[i] != CFG["ignore_index"]:
                            i += 1
                        end = i - 1
                        left  = hidden[b, start - 1] if start > 0 else torch.zeros_like(hidden[b, 0])
                        right = hidden[b, end + 1]   if (end + 1) < hidden.size(1) else torch.zeros_like(hidden[b, 0])
                        left_batch.append(left); right_batch.append(right)
                        rel_pos.append(end - start + 1)
                        targets.append(inp[start])
                    i += 1
                if targets:
                    left_batch = torch.stack(left_batch)
                    right_batch= torch.stack(right_batch)
                    rel_pos   = torch.tensor(rel_pos, device=device)
                    targets   = torch.tensor(targets, device=device)
                    logits = decoder(left_batch, right_batch, rel_pos)
                    loss = loss + criterion(logits, targets)
                    count += 1

            if count > 0:
                loss = loss / count
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                train_batches += 1

        avg_train_loss = total_train_loss / max(train_batches, 1)

        # === Validate ===
        encoder.eval(); decoder.eval()
        total_val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                byte_input = batch['byte_input'].to(device)
                mod_idx    = batch['modality_index'].to(device)
                masked, labels = span_mask_input(
                    byte_input,
                    mask_prob=CFG["mask_prob"],
                    max_span_length=CFG["mask_span_length"],
                    ignore_index=CFG["ignore_index"]
                )
                masked, labels = masked.to(device), labels.to(device)
                hidden = encoder(masked, mod_idx)

                loss = torch.tensor(0.0, device=device)
                count = 0
                B, L, D = hidden.shape
                for b in range(B):
                    inp = byte_input[b]
                    lbl = labels[b]
                    i = 0
                    left_batch, right_batch, rel_pos, targets = [], [], [], []
                    while i < L:
                        if lbl[i] != CFG["ignore_index"]:
                            start = i
                            while i < L and lbl[i] != CFG["ignore_index"]:
                                i += 1
                            end = i - 1
                            left  = hidden[b, start - 1] if start > 0 else torch.zeros_like(hidden[b, 0])
                            right = hidden[b, end + 1]   if (end + 1) < hidden.size(1) else torch.zeros_like(hidden[b, 0])
                            left_batch.append(left); right_batch.append(right)
                            rel_pos.append(end - start + 1)
                            targets.append(inp[start])
                        i += 1
                    if targets:
                        left_batch = torch.stack(left_batch)
                        right_batch= torch.stack(right_batch)
                        rel_pos   = torch.tensor(rel_pos, device=device)
                        targets   = torch.tensor(targets, device=device)
                        logits = decoder(left_batch, right_batch, rel_pos)
                        loss = loss + criterion(logits, targets)
                        count += 1
                if count > 0:
                    total_val_loss += (loss / count).item()
                    val_batches += 1

        avg_val_loss = total_val_loss / max(val_batches, 1)

        # === Log ===
        with open(log_path, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        # === Checkpoint + early stop ===
        if avg_val_loss < best_loss - CFG["early_stopping_delta"]:
            best_loss = avg_val_loss
            no_improve = 0
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, ckpt_path)
            print("✅ Checkpoint saved.")
        else:
            no_improve += 1
            print(f"⚠️ No improvement for {no_improve} epoch(s)")
            if no_improve >= CFG["early_stopping_patience"]:
                print(f"⏹️ Early stopping triggered after {epoch} epochs.")
                break

if __name__ == "__main__":
    main()
