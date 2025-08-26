# train_spanmasked.py — continual pretraining with replay + distillation (+ optional EWC)

import os
import csv
import copy
import argparse
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from configurations.config import config as CFG
from loader.multiModal_dataloader import MultiModalDataset
from utility.span_masking import span_mask_input

# Encoder / Decoder
from encoder.byte_encoder import ByteEncoder
try:
    from decoders.span_boundary_decoder import SpanBoundaryDecoder
except ModuleNotFoundError:
    # fallback if decoder file sits next to this script
    from span_boundary_decoder import SpanBoundaryDecoder

# Continual learning helpers
from trainer.continual import (
    ReplayBuffer, feature_distillation_loss,
    compute_fisher_diag, ewc_penalty
)

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

    fisher = None
    prev_params = None
    old_encoder = None  # frozen snapshot for feature distillation

    if os.path.exists(ckpt_path):
        print(f"\n✅ Loading from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        if "fisher" in ckpt and "prev_params" in ckpt:
            fisher = {k: v.to(device) for k, v in ckpt["fisher"].items()}
            prev_params = {k: v.to(device) for k, v in ckpt["prev_params"].items()}
        # keep a frozen copy for distillation (Learning without Forgetting)
        old_encoder = copy.deepcopy(encoder).eval().requires_grad_(False)
    else:
        print("\n🚨 No checkpoint found — starting from scratch")

    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=CFG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=CFG["ignore_index"])

    # === Data ===
    full_dataset = MultiModalDataset(
        data_path=args.data_path, modality=args.modality, split='train'
    )
    train_size = int(0.9 * len(full_dataset)) if len(full_dataset) > 1 else len(full_dataset)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"], shuffle=False)

    # === Replay Buffer (tiny) ===
    replay = ReplayBuffer(max_per_modality=256)
    modality_name = args.modality

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

            # add to replay (store CPU copy internally)
            # batch["byte_input"] is 1D (L,) per sample when batch_size==1
            # If you raise batch_size, adapt to iterate items.
            replay.add_batch(modality_name, batch["file_path"], batch["byte_input"])

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

            # ----- supervised span-boundary reconstruction -----
            loss_sup = torch.tensor(0.0, device=device)
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
                    loss_sup = loss_sup + criterion(logits, targets)
                    count += 1
            if count > 0:
                loss_sup = loss_sup / count

            # ----- feature distillation on tiny replay (Learning without Forgetting) -----
            loss_distill = torch.tensor(0.0, device=device)
            if old_encoder is not None:
                sample = replay.sample(batch_size=1)
                if sample is not None:
                    _, _, bt = sample[0]  # (L,) tensor (CPU)
                    # Normalize to (1, L) for masking
                    if bt.dim() == 1:
                        bt = bt.unsqueeze(0)
                    masked_r, _ = span_mask_input(
                        bt,
                        mask_prob=CFG["mask_prob"],
                        max_span_length=CFG["mask_span_length"],
                        ignore_index=CFG["ignore_index"]
                    )
                    masked_r = masked_r.to(device)
                    # Use a neutral modality index (0) for feature anchoring
                    with torch.no_grad():
                        old_feat = old_encoder(masked_r, mod_idx.new_zeros(1))
                    new_feat = encoder(masked_r, mod_idx.new_zeros(1))
                    loss_distill = feature_distillation_loss(new_feat, old_feat)

            # ----- optional EWC penalty (device-safe) -----
            loss_ewc = ewc_penalty(encoder, prev_params, fisher, lam=50.0)

            # Combine losses
            loss = loss_sup + 0.5 * loss_distill + loss_ewc
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

                loss_sup = torch.tensor(0.0, device=device)
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
                        loss_sup = loss_sup + criterion(logits, targets)
                        count += 1
                if count > 0:
                    total_val_loss += (loss_sup / count).item()
                    val_batches += 1

        avg_val_loss = total_val_loss / max(val_batches, 1)

        # === Log ===
        with open(log_path, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        # === Checkpoint + Fisher + early stop ===
        if avg_val_loss < best_loss - CFG["early_stopping_delta"]:
            best_loss = avg_val_loss
            no_improve = 0

            # Compute Fisher on a few train samples for next modality’s EWC
            fisher_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            fisher = compute_fisher_diag(encoder, fisher_loader, device, n_steps=200)

            # Snapshot previous params (CPU for portability/size)
            prev_params = {n: p.detach().clone().cpu()
                           for n, p in encoder.named_parameters() if p.requires_grad}

            save_dict = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "fisher":  {k: v.detach().cpu() for k, v in fisher.items()},
                "prev_params": prev_params
            }
            torch.save(save_dict, ckpt_path)
            print("✅ Checkpoint + Fisher saved.")

            # keep live device copies for immediate use this run
            fisher = {k: v.to(device) for k, v in fisher.items()}
            prev_params = {k: v.to(device) for k, v in prev_params.items()}

            # refresh frozen snapshot to the new best
            old_encoder = copy.deepcopy(encoder).eval().requires_grad_(False)

        else:
            no_improve += 1
            print(f"⚠️ No improvement for {no_improve} epoch(s)")
            if no_improve >= patience:
                print(f"⏹️ Early stopping triggered after {epoch} epochs.")
                break

if __name__ == "__main__":
    main()
