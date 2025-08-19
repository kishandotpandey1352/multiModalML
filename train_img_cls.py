# train_img_cls.py
import os, csv, random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix

from data_modules.img_hf import make_loader
from heads.img_classifier_head import ImageClassifierHead

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------- Encoder loader ----------
def load_shared_encoder(init_checkpoint: str, device: str, cfg=None):
    import importlib.util, os, dataclasses
    import torch
    from encoder.byte_encoder import ByteEncoder

    def _obj_to_dict(obj):
        if isinstance(obj, dict):
            return obj
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        d = {}
        for k in dir(obj):
            if k.startswith("_"): 
                continue
            v = getattr(obj, k)
            if isinstance(v, (int, float, bool, str, tuple, list, dict)) or v is None:
                d[k] = v
        return d

    def _normalize_keys(d: dict) -> dict:
        out = dict(d)
        # ByteEncoder expects: embed_dim, max_length, nhead, num_layers, dim_feedforward, dropout
        if "max_length" not in out and "max_len" in out:
            out["max_length"] = out["max_len"]
        if "dim_feedforward" not in out and "dim_feed_forward" in out:
            out["dim_feedforward"] = out["dim_feed_forward"]
        # Provide safe defaults if missing
        out.setdefault("embed_dim",        512)
        out.setdefault("max_length",      1024)
        out.setdefault("nhead",              8)
        out.setdefault("num_layers",         6)
        out.setdefault("dim_feedforward", 2048)
        out.setdefault("dropout",          0.1)
        return out

    tried = []
    cfg_map = None

    # Prefer path-based import (robust on Windows without PYTHONPATH changes)
    if cfg is not None and getattr(cfg, "encoder_config_path", None):
        try:
            path = os.path.abspath(cfg.encoder_config_path)
            spec = importlib.util.spec_from_file_location("enc_cfg_mod", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            cls_name = getattr(cfg, "encoder_config_class", None) or "TrainingConfig"
            Conf = getattr(mod, cls_name, None) or getattr(mod, "TrainingConfig", None) or getattr(mod, "Config", None)
            if Conf is None:
                raise AttributeError(f"No config class named {cls_name}/TrainingConfig/Config in {path}")
            cfg_obj = Conf() if callable(Conf) else Conf
            cfg_map = _normalize_keys(_obj_to_dict(cfg_obj))
        except Exception as e:
            tried.append(f"path {getattr(cfg,'encoder_config_path',None)}:{getattr(cfg,'encoder_config_class',None)}: {e}")

    if cfg_map is None:
        tried.append("no cfg_map produced; using empty dict")
        cfg_map = _normalize_keys({})

    try:
        enc = ByteEncoder(cfg_map)
    except Exception as e:
        tried.append(f"ByteEncoder(cfg_map): {e}")
        msg = "[ERROR] Could not construct ByteEncoder. Attempts:\n" + "\n".join(" - " + t for t in tried)
        raise RuntimeError(msg)

    sd = torch.load(init_checkpoint, map_location="cpu")
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded encoder (strict=False). missing={len(missing)} unexpected={len(unexpected)}")
    enc.eval()
    return enc.to(device)

# ---------- Encoding helper ----------
@torch.no_grad()
def encode_to_sequence(encoder, x, attn, device):
    # Optional defensive truncation if inputs exceed encoder.max_length
    max_len = getattr(encoder, "max_length", x.size(1))
    if x.size(1) > max_len:
        x = x[:, :max_len]
        if attn is not None:
            attn = attn[:, :max_len]

    out = encoder(x.to(device), attention_mask=attn.to(device) if attn is not None else None, return_hidden=True)
    return out if not isinstance(out, tuple) else out[0]  # (B,T,D)

# ---------- Training / Eval ----------
def topk_acc(logits: torch.Tensor, target: torch.Tensor, ks=(1,)) -> list[float]:
    """Return accuracy@k in [0,1] for each k in ks."""
    with torch.no_grad():
        maxk = max(ks)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B, maxk)
        pred = pred.t()  # (maxk, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, B)
        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / target.size(0)).item())
        return res

def train_one_epoch(cfg, encoder, head, loader, device, scaler, optim, epoch):
    encoder.requires_grad_(False)
    head.train()
    ce = nn.CrossEntropyLoss()
    tot, correct, n = 0.0, 0, 0

    for step, batch in enumerate(loader, 1):
        x = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        with torch.cuda.amp.autocast(enabled=cfg.amp and device=='cuda'):
            hidden = encode_to_sequence(encoder, x, attn, device)  # (B,T,D)
            logits = head(hidden, attn)
            loss = ce(logits, y)

        optim.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_grad_norm)
            scaler.step(optim); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_grad_norm)
            optim.step()

        tot += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += y.size(0)

        if step % cfg.log_every_n == 0:
            print(f"[Epoch {epoch}] step {step} loss={tot/max(n,1):.4f} acc={correct/max(n,1):.4f}")

        if getattr(cfg, "max_train_batches", None) and step >= cfg.max_train_batches:
            break

    return tot/max(n,1), correct/max(n,1)

@torch.no_grad()
def evaluate(cfg, encoder, head, loader, device):
    """Legacy eval: returns loss and top-1 accuracy."""
    encoder.eval(); head.eval()
    ce = nn.CrossEntropyLoss()
    tot, correct, n = 0.0, 0, 0
    correct_top5 = 0
    all_preds, all_labels = [], []
    for batch in loader:
        x = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        hidden = encode_to_sequence(encoder, x, attn, device)
        logits = head(hidden, attn)
        loss = ce(logits, y)
        tot += loss.item() * y.size(0)
        # Top-1
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        # Top-5
        top5 = torch.topk(logits, k=5, dim=1).indices
        for i in range(y.size(0)):
            if y[i].item() in top5[i].cpu().tolist():
                correct_top5 += 1
        # collect for confusion matrix
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        n += y.size(0)
    cm = confusion_matrix(all_labels, all_preds)
    return tot/max(n,1), correct/max(n,1), correct_top5/max(n,1), cm

@torch.no_grad()
def evaluate_full(cfg, encoder, head, loader, device, num_classes: int):
    """Extended eval: loss, top1, top5, confusion matrix (num_classes x num_classes)."""
    encoder.eval(); head.eval()
    ce = nn.CrossEntropyLoss()

    tot_loss, n = 0.0, 0
    correct1, correct5 = 0, 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch in loader:
        x = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        hidden = encode_to_sequence(encoder, x, attn, device)
        logits = head(hidden, attn)

        loss = ce(logits, y)
        tot_loss += loss.item() * y.size(0)
        n += y.size(0)

        # Top-1 / Top-5
        t1, t5 = topk_acc(logits, y, ks=(1,5))
        correct1 += int(t1 * y.size(0))
        correct5 += int(t5 * y.size(0))

        # Confusion matrix
        preds = logits.argmax(dim=1)
        for t, p in zip(y.view(-1), preds.view(-1)):
            conf[t.long(), p.long()] += 1

        if getattr(cfg, "max_eval_batches", None) and (conf.sum().item() >= cfg.max_eval_batches * loader.batch_size):
            break

    avg_loss = tot_loss / max(n, 1)
    top1 = correct1 / max(n, 1)
    top5 = correct5 / max(n, 1)
    return avg_loss, top1, top5, conf

def save_confusion_matrix_csv(conf: torch.Tensor, class_names: list[str] | None, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [""] + (class_names if class_names else [f"pred_{i}" for i in range(conf.size(1))])
        w.writerow(header)
        for i in range(conf.size(0)):
            row_name = class_names[i] if class_names else f"true_{i}"
            w.writerow([row_name] + conf[i].tolist())

# ---------- Main ----------
def main(cfg):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_loader(cfg, cfg.split_train, cfg.train_batch_size)
    val_loader   = make_loader(cfg, cfg.split_val,   cfg.val_batch_size)

    encoder = load_shared_encoder(cfg.init_checkpoint, device, cfg)

    # infer hidden size D from a tiny dummy pass
    first = next(iter(train_loader))
    with torch.no_grad():
        seq = encode_to_sequence(encoder, first["input_ids"], first["attention_mask"], device)
        D = seq.size(-1)

    head = ImageClassifierHead(hidden_dim=D, num_classes=cfg.num_classes).to(device)
    optim_head = optim.AdamW(head.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device=='cuda')

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_val = -1.0
    # CSV (epoch log)
    if cfg.csv_log_path:
        os.makedirs(os.path.dirname(cfg.csv_log_path), exist_ok=True)
        with open(cfg.csv_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc"])

    for epoch in range(1, cfg.epochs+1):
        tr_loss, tr_acc = train_one_epoch(cfg, encoder, head, train_loader, device, scaler, optim_head, epoch)
        va_loss, va_acc, va_acc5, cm = evaluate(cfg, encoder, head, val_loader, device)
        print(f"[Epoch {epoch}] train_acc={tr_acc:.4f} val_acc@1={va_acc:.4f} val_acc@5={va_acc5:.4f}")
        print(f"[Epoch {epoch}] Confusion matrix:\n{cm}")

        if cfg.csv_log_path:
            with open(cfg.csv_log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{va_loss:.4f}", f"{va_acc:.4f}", f"{va_acc5:.4f}"])

        metric = va_acc if cfg.best_by == "val_acc" else -va_loss
        if metric > best_val:
            best_val = metric
            torch.save(head.state_dict(), os.path.join(cfg.save_dir, cfg.save_head_as))
            if cfg.save_encoder_as:
                torch.save(encoder.state_dict(), os.path.join(cfg.save_dir, cfg.save_encoder_as))
            print(f"[INFO] Saved best head (val_acc={va_acc:.4f}).")

    # -------- Final
