
import os, random
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from utils.config import TrainingConfig
from encoder.byte_encoder import ByteEncoder
from heads.byte_classifier import ByteClassifierHead

# IMPORTANT: we import HF datasets INSIDE the data module to avoid name collision
from data_modules.agnews_hf import AGNewsHFBytes

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_model(cfg: TrainingConfig):
    enc_cfg = {
        "embed_dim": cfg.embed_dim,
        "max_length": cfg.max_len,
        "nhead": cfg.nhead,
        "dim_feedforward": cfg.dim_feedforward,
        "dropout": cfg.dropout,
        "num_layers": cfg.num_layers,
    }
    encoder = ByteEncoder(enc_cfg)
    head = ByteClassifierHead(d_model=cfg.embed_dim, num_classes=cfg.num_classes, dropout=cfg.dropout)
    return encoder, head

def load_pretrained_encoder(encoder: nn.Module, path: str):
    if not os.path.exists(path):
        print(f"[WARN] init checkpoint not found at {path}. Training encoder from scratch.")
        return
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded encoder (strict=False). missing={len(missing)} unexpected={len(unexpected)}")

def make_loaders(cfg: TrainingConfig):
    full_train = AGNewsHFBytes(split="train", max_len=cfg.max_len, pad_token=cfg.pad_token)
    test_ds    = AGNewsHFBytes(split="test",  max_len=cfg.max_len, pad_token=cfg.pad_token)
    val_size = max(2000, int(cfg.hf_val_split * len(full_train)))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

def evaluate(encoder, head, loader, device):
    encoder.eval(); head.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pooled = encoder(x)               # (B, D)
            logits = head(pooled)             # (B, C)
            loss = ce(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(-1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum/total, correct/total

def train(cfg: TrainingConfig):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, head = build_model(cfg)
    load_pretrained_encoder(encoder, cfg.init_checkpoint)
    encoder.to(device); head.to(device)

    # === NEW: Freeze encoder if requested ===
    if getattr(cfg, "freeze_encoder", False):
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()  # disable dropout in frozen parts

    train_loader, val_loader, test_loader = make_loaders(cfg)

    # === NEW: Optimizer only on head when encoder is frozen ===
    if getattr(cfg, "freeze_encoder", False):
        params = [{"params": head.parameters(), "lr": cfg.lr}]
    else:
        params = [
            {"params": encoder.parameters(), "lr": cfg.lr},
            {"params": head.parameters(),    "lr": cfg.lr},
        ]
    optim = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device=="cuda")

    best_val = 0.0
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs+1):
        # === Train ===
        if not getattr(cfg, "freeze_encoder", False):
            encoder.train()
        else:
            encoder.eval()  # keep frozen encoder in eval mode
        head.train()

        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device=="cuda"):
                pooled = encoder(x)
                logits = head(pooled)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
                if not getattr(cfg, "freeze_encoder", False):
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(-1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = loss_sum/total; train_acc = correct/total

        # === Validate ===
        val_loss, val_acc = evaluate(encoder, head, val_loader, device)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        # === NEW: Conditional checkpoint saving ===
        enc_path = os.path.join(cfg.checkpoints_dir, cfg.save_encoder_as)
        head_path = os.path.join(cfg.checkpoints_dir, cfg.save_head_as)

        # Save rolling checkpoints
        if not getattr(cfg, "freeze_encoder", False):
            torch.save(encoder.state_dict(), enc_path)
        torch.save(head.state_dict(), head_path)

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            if not getattr(cfg, "freeze_encoder", False):
                torch.save(encoder.state_dict(), os.path.join(cfg.checkpoints_dir, "best_" + cfg.save_encoder_as))
            torch.save(head.state_dict(), os.path.join(cfg.checkpoints_dir, "best_" + cfg.save_head_as))
            print(f"[Best] val_acc={best_val:.4f} — saved best checkpoints")

    # === Test ===
    test_loss, test_acc = evaluate(encoder, head, test_loader, device)
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f}")

def main(cfg: TrainingConfig | None = None):
    if cfg is None:
        cfg = TrainingConfig()
    train(cfg)

if __name__ == "__main__":
    main()
