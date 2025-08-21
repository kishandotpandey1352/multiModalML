# train_tab_bin.py
import os, csv, random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from utils.config_tab_bin import TabBinConfig
from data_modules.tab_hf import make_loader
from heads.tab_binary_head import TabBinaryHead

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

# ----- Shared encoder loader (matches your image scripts) -----
def load_shared_encoder(init_checkpoint: str, device: str, cfg=None):
    import importlib.util, os, dataclasses
    from encoder.byte_encoder import ByteEncoder

    def _obj_to_dict(obj):
        if isinstance(obj, dict): return obj
        if dataclasses.is_dataclass(obj): return dataclasses.asdict(obj)
        d = {}
        for k in dir(obj):
            if k.startswith("_"): continue
            v = getattr(obj, k)
            if isinstance(v, (int,float,bool,str,tuple,list,dict)) or v is None:
                d[k] = v
        return d

    def _normalize_keys(d: dict) -> dict:
        out = dict(d)
        if "max_length" not in out and "max_len" in out: out["max_length"] = out["max_len"]
        if "dim_feedforward" not in out and "dim_feed_forward" in out: out["dim_feedforward"] = out["dim_feed_forward"]
        out.setdefault("embed_dim",        512)
        out.setdefault("max_length",      1024)
        out.setdefault("nhead",              8)
        out.setdefault("num_layers",         6)
        out.setdefault("dim_feedforward", 2048)
        out.setdefault("dropout",          0.1)
        return out

    tried = []
    cfg_map = None
    if cfg is not None and getattr(cfg, "encoder_config_path", None):
        try:
            path = os.path.abspath(cfg.encoder_config_path)
            spec = importlib.util.spec_from_file_location("enc_cfg_mod", path)
            mod = importlib.util.module_from_spec(spec); assert spec.loader is not None
            spec.loader.exec_module(mod)
            cls_name = getattr(cfg, "encoder_config_class", None) or "TrainingConfig"
            Conf = getattr(mod, cls_name, None) or getattr(mod, "TrainingConfig", None) or getattr(mod, "Config", None)
            if Conf is None: raise AttributeError(f"No config class {cls_name}/TrainingConfig/Config in {path}")
            cfg_obj = Conf() if callable(Conf) else Conf
            cfg_map = _normalize_keys(_obj_to_dict(cfg_obj))
        except Exception as e:
            tried.append(f"path {getattr(cfg,'encoder_config_path',None)}:{getattr(cfg,'encoder_config_class',None)}: {e}")
    if cfg_map is None:
        cfg_map = _normalize_keys({})

    enc = ByteEncoder(cfg_map)
    sd = torch.load(init_checkpoint, map_location="cpu")
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded encoder (strict=False). missing={len(missing)} unexpected={len(unexpected)}")
    enc.eval()
    return enc.to(device)

@torch.no_grad()
def encode_to_sequence(encoder, x, attn, device):
    max_len = getattr(encoder, "max_length", x.size(1))
    if x.size(1) > max_len:
        x = x[:, :max_len]
        if attn is not None:
            attn = attn[:, :max_len]
        if not hasattr(encode_to_sequence, "_warned"):
            print(f"[WARN] Truncating from {x.size(1)} to encoder.max_length={max_len}.")
            encode_to_sequence._warned = True
    out = encoder(x.to(device), attention_mask=attn.to(device) if attn is not None else None, return_hidden=True)
    return out if not isinstance(out, tuple) else out[0]  # (B,T,D)

def evaluate(cfg, encoder, head, loader, device, bce):
    encoder.eval(); head.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["label"].to(device)  # float
            seq = encode_to_sequence(encoder, x, attn, device)
            logits = head(seq, attn)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits, 0).numpy()
    labels = torch.cat(all_labels, 0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int32)

    # metrics
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    f1 = f1_score(labels, preds) if len(np.unique(labels)) == 2 else float("nan")
    acc = accuracy_score(labels, preds)
    loss = float(bce(torch.tensor(logits), torch.tensor(labels)).item())
    return {"auc": auc, "f1": f1, "acc": acc, "loss": loss}

def train_one_epoch(cfg, encoder, head, loader, device, scaler, optim, epoch):
    encoder.requires_grad_(False)
    head.train()
    bce = nn.BCEWithLogitsLoss()
    running, n_seen = 0.0, 0
    for step, batch in enumerate(loader, 1):
        x = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["label"].to(device)  # float

        with torch.cuda.amp.autocast(enabled=cfg.amp and device == 'cuda'):
            seq = encode_to_sequence(encoder, x, attn, device)
            logits = head(seq, attn)
            loss = bce(logits, y)

        optim.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_grad_norm)
            scaler.step(optim); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_grad_norm)
            optim.step()

        running += loss.item() * y.size(0)
        n_seen += y.size(0)

        if step % cfg.log_every_n == 0:
            print(f"[Epoch {epoch}] step {step} loss={running/max(n_seen,1):.4f}")

        if cfg.max_train_batches and step >= cfg.max_train_batches:
            break
    return running / max(n_seen, 1)

def main(cfg: TabBinConfig):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_loader(cfg, cfg.split_train, cfg.train_batch_size)
    val_loader   = make_loader(cfg, cfg.split_val,   cfg.val_batch_size)

    encoder = load_shared_encoder(cfg.init_checkpoint, device, cfg)

    # infer hidden size
    first = next(iter(train_loader))
    with torch.no_grad():
        D = encode_to_sequence(encoder, first["input_ids"], first["attention_mask"], device).size(-1)

    head = TabBinaryHead(hidden_dim=D).to(device)
    optim_head = optim.AdamW(head.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device=='cuda')
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(cfg.save_dir, exist_ok=True)
    if cfg.csv_log_path:
        os.makedirs(os.path.dirname(cfg.csv_log_path), exist_ok=True)
        with open(cfg.csv_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","val_auc","val_f1","val_acc"])

    best = -1e9
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(cfg, encoder, head, train_loader, device, scaler, optim_head, epoch)
        val = evaluate(cfg, encoder, head, val_loader, device, bce)
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} | val_auc={val['auc']:.4f} f1={val['f1']:.4f} acc={val['acc']:.4f}")

        if cfg.csv_log_path:
            with open(cfg.csv_log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{val['loss']:.4f}", f"{val['auc']:.4f}", f"{val['f1']:.4f}", f"{val['acc']:.4f}"])

        # pick best by AUC (fallback to accuracy when AUC is nan)
        score = val["auc"] if not (val["auc"] != val["auc"]) else val["acc"]
        if score > best:
            best = score
            torch.save(head.state_dict(), os.path.join(cfg.save_dir, cfg.save_head_as))
            print(f"[INFO] Saved best head (score={score:.4f}).")

if __name__ == "__main__":
    main(TabBinConfig())
