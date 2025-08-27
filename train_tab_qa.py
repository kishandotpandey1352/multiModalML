# train_tab_qa.py
import os, time, random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import re
import pandas as pd
from itertools import islice

from utils.config_tab_qa import TabQAConfig
from data_modules.tabqa_synth import make_loader
from heads.tabqa_span_head import TabQASpanHead
import math
from torch.optim.lr_scheduler import LambdaLR
# ------------------------------
# AMP compatibility shim (PyTorch <2.0 and >=2.0)
# ------------------------------
try:
    # New API (PyTorch ≥ 2.0)
    from torch.amp import autocast as _autocast_new
    from torch.amp import GradScaler as _GradScaler_new
    def autocast_ctx(enabled: bool, device_type: str = "cuda"):
        return _autocast_new(device_type=device_type, enabled=enabled)
    GradScaler = _GradScaler_new
except Exception:
    # Old API (PyTorch < 2.0)
    from torch.cuda.amp import autocast as _autocast_old
    from torch.cuda.amp import GradScaler as _GradScaler_old
    def autocast_ctx(enabled: bool, device_type: str = "cuda"):
        # old API ignores device_type and only supports CUDA
        return _autocast_old(enabled=enabled)
    GradScaler = _GradScaler_old

# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _nice_shape(v):
    if isinstance(v, torch.Tensor):
        return f"tensor{tuple(v.shape)} {v.dtype}"
    if isinstance(v, (list, tuple)):
        return f"list(len={len(v)})"
    return type(v).__name__

def to_device_batch(batch, device: str):
    """Move only tensors to device; leave strings/lists on CPU (e.g., answer_text)."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def _unfreeze_last_n_transformer_layers(model: torch.nn.Module, n: int) -> int:
    """
    Try to unfreeze the last N transformer blocks based on common name patterns.
    Returns how many parameter tensors were unfrozen.
    """
    if n <= 0:
        return 0
    patterns = [re.compile(r'\blayers\.(\d+)\.'),
                re.compile(r'\bencoder\.layers\.(\d+)\.'),
                re.compile(r'\btransformer\.layers\.(\d+)\.')]
    # discover layer indices
    idxs = set()
    for name, _ in model.named_parameters():
        for p in patterns:
            m = p.search(name)
            if m:
                idxs.add(int(m.group(1)))
    if not idxs:
        return 0
    top = sorted(idxs)[-n:]
    unfrozen = 0
    for name, p in model.named_parameters():
        if any(f'.layers.{i}.' in name for i in top) or \
           any(f'encoder.layers.{i}.' in name for i in top) or \
           any(f'transformer.layers.{i}.' in name for i in top):
            p.requires_grad_(True); unfrozen += 1
    return unfrozen


# ------------------------------
# Encoder loader (your pattern)
# ------------------------------
def load_shared_encoder(init_checkpoint: str, device: str, cfg: Optional[TabQAConfig]=None):
    import importlib.util, dataclasses
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
        out.setdefault("embed_dim", 512)
        out.setdefault("max_length", 1024)
        out.setdefault("nhead", 8)
        out.setdefault("num_layers", 6)
        out.setdefault("dim_feedforward", 2048)
        out.setdefault("dropout", 0.1)
        return out

    cfg_map = {}
    if cfg and cfg.encoder_config_path:
        path = os.path.abspath(cfg.encoder_config_path)
        spec = importlib.util.spec_from_file_location("enc_cfg_mod", path)
        mod = importlib.util.module_from_spec(spec); assert spec.loader is not None
        spec.loader.exec_module(mod)
        cls_name = cfg.encoder_config_class or "TrainingConfig"
        Conf = getattr(mod, cls_name, None) or getattr(mod, "TrainingConfig", None) or getattr(mod, "Config", None)
        cfg_obj = Conf() if callable(Conf) else Conf
        cfg_map = _normalize_keys(_obj_to_dict(cfg_obj))
    else:
        cfg_map = _normalize_keys({})

    enc = ByteEncoder(cfg_map)
    sd = torch.load(init_checkpoint, map_location="cpu")
    enc.load_state_dict(sd, strict=False)
    enc.eval().to(device)
    return enc

@torch.no_grad()
def encode_to_sequence(encoder, x, attn, device):
    """Ensure inputs are truncated to encoder.max_length and returned on 'device'."""
    max_len = getattr(encoder, "max_length", x.size(1))
    if x.size(1) > max_len:
        x = x[:, :max_len]
        if attn is not None:
            attn = attn[:, :max_len]
        if not hasattr(encode_to_sequence, "_warned"):
            print(f"[WARN] Truncating inputs to encoder.max_length={max_len}")
            encode_to_sequence._warned = True
    out = encoder(x.to(device), attention_mask=attn.to(device) if attn is not None else None, return_hidden=True)
    seq = out if not isinstance(out, tuple) else out[0]  # (B,T,D)
    return seq  # should already be on device

def char_f1(pred: str, gold: str) -> float:
    from collections import Counter
    pc, gc = Counter(pred), Counter(gold)
    common = sum((pc & gc).values())
    if common == 0: return 0.0
    p = common / max(1, sum(pc.values()))
    r = common / max(1, sum(gc.values()))
    return 2 * p * r / (p + r + 1e-12)

# ------------------------------
# Eval / Train
# ------------------------------
def evaluate(cfg, encoder, head, loader, device):
    encoder.eval()
    head.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss, nitems = 0.0, 0
    all_em, all_f1 = [], []
    with torch.no_grad():
        for batch in islice(loader, cfg.val_steps):
            batch = to_device_batch(batch, device)
            x = batch["input_ids"]
            m = batch["attention_mask"]
            s_idx = batch["start_idx"]
            e_idx = batch["end_idx"]

            seq = encode_to_sequence(encoder, x, m, device)     # [B,T,D] on device
            start_logits, end_logits = head(seq, m)              # [B,T],[B,T]
            # Constrain logits to column-token window if provided
            if "win_start" in batch and "win_end" in batch:
                T = start_logits.size(1)
                ar = torch.arange(T, device=start_logits.device).unsqueeze(0)  # [1,T]
                ws = batch["win_start"].unsqueeze(1)  # [B,1]
                we = batch["win_end"].unsqueeze(1)    # [B,1]
                win_mask = (ar >= ws) & (ar <= we)    # True inside allowed window
                # Also respect attention_mask (ignore pads)
                if "attention_mask" in batch:
                    win_mask = win_mask & (batch["attention_mask"].bool())
                start_logits = start_logits.masked_fill(~win_mask, -1e9)
                end_logits   = end_logits.masked_fill(~win_mask, -1e9)

            loss = ce(start_logits, s_idx) + ce(end_logits, e_idx)
            tot_loss += loss.item() * x.size(0)
            nitems += x.size(0)

            # decode predictions to text (bytes → utf-8)
            start_pred = start_logits.argmax(dim=-1)
            end_pred   = end_logits.argmax(dim=-1)
            end_pred = torch.maximum(end_pred, start_pred)

            for i in range(x.size(0)):
                ids = x[i].detach().cpu().tolist()
                real_len = int(m[i].sum().item())
                bb = bytes([(cfg.remap_255_to if t == cfg.pad_token else t) for t in ids[:real_len]])
                s, e = int(start_pred[i]), int(end_pred[i])
                s = max(0, min(s, real_len - 1)); e = max(0, min(e, real_len - 1))
                pred_txt = bb[s:e+1].decode("utf-8", "ignore")
                gold_txt = batch["answer_text"][i]
                em = 1.0 if pred_txt.strip() == str(gold_txt).strip() else 0.0
                f1 = char_f1(pred_txt.strip(), str(gold_txt).strip())
                all_em.append(em); all_f1.append(f1)

    avg_loss = tot_loss / max(1, nitems)
    return {"val_loss": avg_loss,
            "val_em": float(np.mean(all_em) if all_em else 0.0),
            "val_f1": float(np.mean(all_f1) if all_f1 else 0.0)}

def train_epoch(cfg, encoder, head, loader, device, scaler, optim, scheduler, global_step):
    if any(p.requires_grad for p in encoder.parameters()):
        encoder.train()
    else:
        encoder.eval()
    head.train()
    ce = nn.CrossEntropyLoss()
    running, seen = 0.0, 0
    last_log = time.time()

    for step, batch in enumerate(islice(loader, cfg.train_steps_per_epoch), 1):
        t_fetch = time.time()
        batch = to_device_batch(batch, device)
        fetch_dt = time.time() - t_fetch
        x = batch["input_ids"]
        m = batch["attention_mask"]
        s_idx = batch["start_idx"]
        e_idx = batch["end_idx"]

        t_fw = time.time()
        with autocast_ctx(cfg.amp and device == "cuda", device_type="cuda"):
            seq = encode_to_sequence(encoder, x, m, device)
            start_logits, end_logits = head(seq, m)
            # Constrain logits to column-token window if provided
            if "win_start" in batch and "win_end" in batch:
                T = start_logits.size(1)
                ar = torch.arange(T, device=start_logits.device).unsqueeze(0)  # [1,T]
                ws = batch["win_start"].unsqueeze(1)  # [B,1]
                we = batch["win_end"].unsqueeze(1)    # [B,1]
                win_mask = (ar >= ws) & (ar <= we)    # True inside allowed window
                # Also respect attention_mask (ignore pads)
                if "attention_mask" in batch:
                    win_mask = win_mask & (batch["attention_mask"].bool())
                start_logits = start_logits.masked_fill(~win_mask, -1e9)
                end_logits   = end_logits.masked_fill(~win_mask, -1e9)
                
            loss = ce(start_logits, s_idx) + ce(end_logits, e_idx)
        fw_dt = time.time() - t_fw

        optim.zero_grad(set_to_none=True)
        if scaler is not None and (cfg.amp and device == "cuda"):
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_grad_norm)
            scaler.step(optim); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_grad_norm)
            optim.step()

        # <-- advance LR schedule per batch
        scheduler.step()
        global_step += 1

        running += loss.item() * x.size(0); seen += x.size(0)

        if device == "cuda" and step == 1:
            print(f"[gpu] seq.device={seq.device}  "
                  f"alloc={torch.cuda.memory_allocated()/1e6:.1f}MB  "
                  f"reserved={torch.cuda.memory_reserved()/1e6:.1f}MB")

        if step % cfg.log_every_n == 0:
            dt = time.time() - last_log; last_log = time.time()
            cur_loss = running / max(1, seen)
            # print(f"[train] step {step} loss={cur_loss:.4f} dt={dt:.2f}s seen={seen}")
            print(f"[train] step {step} fetch={fetch_dt*1000:.1f}ms fw={fw_dt*1000:.1f}ms loss={running/max(1,seen):.4f}")
            lr_now = optim.param_groups[0]["lr"]
            print(f"[train] step {step} loss={cur_loss:.4f} lr={lr_now:.2e} dt={dt:.2f}s seen={seen}")




    return running / max(1, seen), global_step

# ------------------------------
# Main
# ------------------------------
def main(cfg: TabQAConfig, dataframe: Optional["pd.DataFrame"]=None):
    os.makedirs(cfg.save_dir, exist_ok=True)
    print("[cfg]", cfg)
    print("[paths] save_dir:", os.path.abspath(cfg.save_dir))
    print("[paths] csv_log_path:", os.path.abspath(cfg.csv_log_path) if getattr(cfg, "csv_log_path", None) else None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] selected={device}  available={torch.cuda.is_available()}")
    if device == "cuda":
        print(f"[cuda] visible={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
        print(f"[cuda] count={torch.cuda.device_count()}  name={torch.cuda.get_device_name(0)}")

    if cfg.csv_log_path:
        os.makedirs(os.path.dirname(cfg.csv_log_path), exist_ok=True)

    set_seed(cfg.seed)

    # Data
    train_loader = make_loader(cfg, "train", cfg.train_batch_size, dataframe=dataframe)
    val_loader   = make_loader(cfg, "val",   cfg.val_batch_size,   dataframe=dataframe)

    # Encoder
    encoder = load_shared_encoder(cfg.init_checkpoint, device, cfg)
    # Freeze-all, then optionally unfreeze top-N blocks
    for p in encoder.parameters():
        p.requires_grad_(False)
    n_unfrozen = _unfreeze_last_n_transformer_layers(encoder, getattr(cfg, "unfreeze_n_layers", 0))
    print(f"[encoder] unfreeze_n_layers={getattr(cfg, 'unfreeze_n_layers', 0)}  params_unfrozen={n_unfrozen}")
    encoder.train() if n_unfrozen > 0 else encoder.eval()
    
    # Infer hidden dim from a real batch
    print("[setup] building first batch to infer hidden dim...", flush=True)
    first = next(iter(train_loader))
    if getattr(cfg, "debug_label_check", False):
        for k in range(min(5, first["input_ids"].shape[0])):
            ids = first["input_ids"][k].tolist()
            mask_len = int(first["attention_mask"][k].sum().item())
            bb = bytes([(cfg.remap_255_to if t==cfg.pad_token else t) for t in ids[:mask_len]])
            s, e = int(first["start_idx"][k]), int(first["end_idx"][k])
            gold_span = bb[max(0,s):min(mask_len, e+1)].decode("utf-8","ignore")
            print(f"[label-check] {k}: gold='{first['answer_text'][k]}' span='{gold_span}' s={s} e={e}")
    
    print("[train] first batch:", {k: _nice_shape(v) for k, v in first.items()})
    with torch.no_grad():
        D = encode_to_sequence(encoder,
                               torch.as_tensor(first["input_ids"]),
                               torch.as_tensor(first["attention_mask"]),
                               device).size(-1)
    print(f"[model] hidden_dim D={D}")
    print("[setup] starting training...", flush=True)

    # Head & optim
    # Head & optim (optimize head + any unfrozen encoder params)
    head = TabQASpanHead(hidden_dim=D).to(device)

    trainable = list(head.parameters()) + [p for p in encoder.parameters() if p.requires_grad]
    # a slightly smaller LR is safer when encoder is partially unfrozen
    lr = min(cfg.lr_head, 1e-3) if any(p.requires_grad for p in encoder.parameters()) else cfg.lr_head
    optim_head = optim.AdamW(trainable, lr=lr, weight_decay=cfg.weight_decay)

    scaler = GradScaler(enabled=(cfg.amp and device == "cuda"))
    # ---- LR scheduler: 10% warmup, then cosine decay over all steps ----
    warmup = int(0.1 * cfg.epochs * cfg.train_steps_per_epoch)
    total  = max(1, cfg.epochs * cfg.train_steps_per_epoch)

    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        p = float(step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))  # cosine

    scheduler = LambdaLR(optim_head, lr_lambda)
    global_step = 0
    # CSV log
    if cfg.csv_log_path:
        import csv
        with open(cfg.csv_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","val_em","val_f1"])

    # Train loop
    best_score = -1.0
    for epoch in range(1, cfg.epochs + 1):
        tr, global_step = train_epoch(cfg, encoder, head, train_loader, device, scaler, optim_head, scheduler, global_step)
        val = evaluate(cfg, encoder, head, val_loader, device)
        print(f"[epoch {epoch}] train_loss={tr:.4f}  val_loss={val['val_loss']:.4f}  "
              f"EM={val['val_em']:.3f}  F1={val['val_f1']:.3f}")

        # CSV append
        if cfg.csv_log_path:
            import csv
            with open(cfg.csv_log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{tr:.4f}", f"{val['val_loss']:.4f}",
                                        f"{val['val_em']:.4f}", f"{val['val_f1']:.4f}"])

        # Save best by EM
        score = val["val_em"]
        if score > best_score:
            best_score = score
            best_path = os.path.join(cfg.save_dir, cfg.save_head_as or "tab_qa_span_head.pth")
            torch.save(head.state_dict(), best_path)
            print(f"[ckpt] Saved best span head → {best_path}")

    # Always save a last checkpoint (safety)
    last_path = os.path.join(cfg.save_dir, "last_head.pth")
    torch.save(head.state_dict(), last_path)
    print(f"[ckpt] Saved last span head → {last_path}")

if __name__ == "__main__":
    main(TabQAConfig())
