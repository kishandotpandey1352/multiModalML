
import os, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from encoder.byte_encoder import ByteEncoder
from models.byte_classifier import ByteClassifier
from data_modules.agnews_hf_bytes import AGNewsBytesHF
from utils.config import get_config

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_pretrained_encoder(encoder: torch.nn.Module, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path} -> training from scratch")
        return
    state = torch.load(ckpt_path, map_location="cpu")
    # Allow both raw state_dict or wrapped
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"[Load] strict=False | missing={len(missing)} unexpected={len(unexpected)}")

def main():
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() and cfg.get("device","cuda")=="cuda" else "cpu"
    set_seed(42)

    # Build encoder with your exact constructor signature
    enc = ByteEncoder({
        "embed_dim": cfg["embed_dim"],
        "max_length": cfg["seq_len"],
        "nhead": cfg["nhead"],
        "dim_feedforward": cfg["dim_feedforward"],
        "dropout": cfg["dropout"],
        "num_layers": cfg["num_layers"],
    })
    load_pretrained_encoder(enc, cfg["checkpoint_path"])

    model = ByteClassifier(enc, num_classes=cfg["num_classes"]).to(device)

    # Data from HF with byte padding=255
    full_train = AGNewsBytesHF(split="train", max_len=cfg["max_len"], pad_token=cfg["pad_token"])
    test_ds    = AGNewsBytesHF(split="test",  max_len=cfg["max_len"], pad_token=cfg["pad_token"])

    # carve out val split (5%)
    val_size = max(2000, int(0.05 * len(full_train)))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Optimizer / Loss
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    ce = nn.CrossEntropyLoss()

    best_val = 0.0
    epochs = cfg["epochs"]
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    for epoch in range(1, epochs+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()

            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = loss_sum/total; train_acc = correct/total

        # eval
        def evaluate(loader):
            model.eval()
            total, correct, loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device); y = y.to(device)
                    logits = model(x)
                    loss = ce(logits, y)
                    loss_sum += loss.item() * x.size(0)
                    preds = logits.argmax(-1)
                    correct += (preds == y).sum().item()
                    total += x.size(0)
            return loss_sum/total, correct/total

        val_loss, val_acc = evaluate(val_loader)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"encoder": model.encoder.state_dict(),
                        "classifier": model.state_dict()},
                       "checkpoints/agnews_classifier_best.pth")
            print(f"[Best] val_acc={best_val:.4f} -> saved checkpoints/agnews_classifier_best.pth")

    # Final test
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in test_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += x.size(0)
        test_loss = loss_sum/total; test_acc = correct/total
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f}")

    # Save final
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict(),
                "classifier": model.state_dict()},
               "checkpoints/agnews_classifier_ft.pth")
    print("Saved final fine-tuned model to checkpoints/agnews_classifier_ft.pth")

if __name__ == "__main__":
    main()
