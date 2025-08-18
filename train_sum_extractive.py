
import os, random
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config_sum_ext import SumExtConfig
from utils.logger import CSVLogger
from data_modules.sum_extractive_hf import XSumExtractiveIterable
from heads.sent_selector_head import SentenceSelectorHead
from encoder.byte_encoder import ByteEncoder  # your shared encoder
from utils.rouge_l import rouge_l_f1

# -------------------- AMP compatibility shim --------------------
try:
    # PyTorch >= 2.0
    from torch.amp import GradScaler as _GradScaler
    from torch import amp as _amp_mod
    def _autocast(enabled: bool):
        return _amp_mod.autocast('cuda', enabled=enabled)
    _AMP_AVAILABLE = True
except Exception:
    try:
        # Older PyTorch (CUDA AMP)
        from torch.cuda.amp import GradScaler as _GradScaler
        from torch.cuda.amp import autocast as _autocast_ctx
        def _autocast(enabled: bool):
            return _autocast_ctx(enabled=enabled)
        _AMP_AVAILABLE = True
    except Exception:
        # No AMP available
        _GradScaler = None
        def _autocast(enabled: bool):
            class _Dummy:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc_val, exc_tb): return False
            return _Dummy()
        _AMP_AVAILABLE = False
# ---------------------------------------------------------------

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_pretrained_encoder(encoder: nn.Module, path: str):
    if not os.path.exists(path):
        print(f"[WARN] init checkpoint not found at {path}. Training encoder from scratch.")
        return
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded encoder (strict=False). missing={len(missing)} unexpected={len(unexpected)}")

def build_model(cfg: SumExtConfig):
    enc_cfg = {
        "embed_dim": cfg.embed_dim,
        "max_length": cfg.src_max_len,
        "nhead": cfg.nhead,
        "dim_feedforward": cfg.dim_feedforward,
        "dropout": cfg.dropout,
        "num_layers": cfg.num_layers,
    }
    encoder = ByteEncoder(enc_cfg)
    head = SentenceSelectorHead(d_model=cfg.embed_dim, dropout=cfg.dropout)
    return encoder, head

def encode_to_sequence(encoder: nn.Module, x: torch.Tensor, attn: Optional[torch.Tensor] = None):
    if attn is not None and hasattr(attn, "device") and attn.device != x.device:
        attn = attn.to(x.device, non_blocking=True)
    # Try forward(..., return_hidden=True)
    try:
        out = encoder(x, attention_mask=attn, return_hidden=True)
        hidden = out[0] if isinstance(out, tuple) else out
        if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
            return hidden
    except TypeError:
        pass
    # Try forward_hidden
    if hasattr(encoder, "forward_hidden"):
        try:
            hidden = encoder.forward_hidden(x, attention_mask=attn)
            if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
                return hidden
        except TypeError:
            hidden = encoder.forward_hidden(x)
            if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
                return hidden
    # Fallback
    out = encoder(x)
    if isinstance(out, torch.Tensor) and out.dim() == 3:
        return out
    raise RuntimeError("Encoder must return (B,T,D) hidden states.")

def mean_pool_spans(hidden: torch.Tensor, spans: torch.Tensor, sent_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden: (B,T,D)
    spans: (B,S,2) start,end (exclusive), -1 padded
    sent_mask: (B,S) binary
    returns: (B,S,D)
    """
    B, T, D = hidden.shape
    _, S, _ = spans.shape
    embs = hidden.new_zeros(B, S, D)
    for b in range(B):
        for s in range(S):
            if sent_mask[b, s].item() == 0:
                continue
            st, en = spans[b, s].tolist()
            if st < 0 or en <= st or st >= T:
                continue
            en = min(en, T)
            seg = hidden[b, st:en]  # (L,D)
            if seg.size(0) == 0:
                continue
            embs[b, s] = seg.mean(dim=0)
    return embs

def evaluate(encoder, head, loader, device, cfg: SumExtConfig) -> Dict[str, float]:
    encoder.eval(); head.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_sent = 0
    rouge_sum = 0.0
    n_docs = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)           # (B,S) float
            am = batch["attention_mask"].to(device)
            spans = batch["spans"].to(device)
            smask = batch["sent_mask"].to(device)

            hidden = encode_to_sequence(encoder, x, am)         # (B,T,D)
            sent_embs = mean_pool_spans(hidden, spans, smask)   # (B,S,D)
            logits = head(sent_embs)                            # (B,S)
            loss = bce(logits[smask==1], y[smask==1])
            total_loss += loss.item()
            total_sent += int(smask.sum().item())

            # ROUGE-L on extracted summaries
            probs = torch.sigmoid(logits)
            topk = cfg.top_k
            for b in range(x.size(0)):
                valid = smask[b].bool()
                k = min(int(valid.sum().item()), topk)
                if k == 0:
                    continue
                scores = probs[b][valid]
                idxs = torch.argsort(scores, descending=True)[:k]
                sel_positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)[idxs].cpu().tolist()
                preds = []
                for s in sel_positions:
                    st, en = spans[b, s].tolist()
                    if st < 0 or en <= st: 
                        continue
                    by = x[b, st:en].detach().cpu().tolist()
                    try:
                        preds.append(bytes([t for t in by if t != 255]).decode("utf-8", errors="ignore"))
                    except Exception:
                        continue
                pred_summary = " ".join(preds)
                ref_summary = batch["gold_summary"][b]
                rouge_sum += rouge_l_f1(pred_summary, ref_summary)
                n_docs += 1

    avg_loss = total_loss / max(total_sent, 1)
    avg_rouge = rouge_sum / max(n_docs, 1)
    return {"loss": avg_loss, "rougeL": avg_rouge}

def train(cfg: SumExtConfig):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Datasets & loaders (IterableDataset; keep num_workers=0)
    def make_loader(split, batch_size):
        ds = XSumExtractiveIterable(
            split=split,
            dataset_name=cfg.dataset_name,
            dataset_revision=cfg.dataset_revision,
            use_hf_streaming=cfg.use_hf_streaming,
            verification_no_checks=getattr(cfg, 'verification_no_checks', True),
            src_field=cfg.src_field,
            tgt_field=cfg.tgt_field,
            src_max_len=cfg.src_max_len,
            pad_token=cfg.pad_token,
            add_space_between_sentences=cfg.add_space_between_sentences,
            max_sentences=cfg.max_sentences,
            top_k=cfg.top_k,
            max_summary_tokens=cfg.max_summary_tokens,
            oracle_metric=getattr(cfg, 'oracle_metric', 'rouge1'),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    train_loader = make_loader(cfg.split_train, cfg.train_batch_size)
    val_loader   = make_loader(cfg.split_val,   cfg.eval_batch_size)
    test_loader  = make_loader(cfg.split_test,  cfg.eval_batch_size)

    # Build model
    encoder, head = build_model(cfg)
    load_pretrained_encoder(encoder, cfg.init_checkpoint)
    encoder.to(device); head.to(device)

    # Freeze encoder if requested
    if cfg.freeze_encoder:
        for p in encoder.parameters(): p.requires_grad = False
        encoder.eval()

    # Optim / scaler
    params = [{"params": head.parameters(), "lr": cfg.lr_head}]
    if not cfg.freeze_encoder:
        params.append({"params": encoder.parameters(), "lr": cfg.lr_encoder})
    optim = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    use_amp = bool(cfg.amp and device == "cuda" and _AMP_AVAILABLE and _GradScaler is not None)
    scaler = _GradScaler(enabled=use_amp) if _GradScaler is not None else None

    # Logger & ckpts
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    logger = CSVLogger(cfg.csv_log_path, fieldnames=["epoch","split","loss","rougeL"])
    best_metric = -1e9 if cfg.save_best_by != "loss" else 1e9

    for epoch in range(1, cfg.epochs+1):
        # ---- Train ----
        if not cfg.freeze_encoder: encoder.train()
        head.train()

        tot_loss = 0.0
        tot_sent = 0
        for step, batch in enumerate(train_loader, 1):
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)           # (B,S) float
            am = batch["attention_mask"].to(device, non_blocking=True)
            spans = batch["spans"].to(device, non_blocking=True)
            smask = batch["sent_mask"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            if use_amp:
                with _autocast(True):
                    hidden = encode_to_sequence(encoder, x, am)           # (B,T,D)
                    sent_embs = mean_pool_spans(hidden, spans, smask)     # (B,S,D)
                    logits = head(sent_embs)                              # (B,S)
                    loss = bce(logits[smask==1], y[smask==1])
                scaler.scale(loss).backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
                    if not cfg.freeze_encoder:
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                hidden = encode_to_sequence(encoder, x, am)           # (B,T,D)
                sent_embs = mean_pool_spans(hidden, spans, smask)     # (B,S,D)
                logits = head(sent_embs)                              # (B,S)
                loss = bce(logits[smask==1], y[smask==1])
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
                    if not cfg.freeze_encoder:
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
                optim.step()

            tot_loss += loss.item()
            tot_sent += int(smask.sum().item())

            # periodic progress logging and optional early stop
            if step % getattr(cfg, 'log_every_n', 200) == 0:
                bs = int(batch['sent_mask'].size(0))
                print(f"[Epoch {epoch}] step {step} (bs={bs}) running_loss={tot_loss/max(tot_sent,1):.4f}")
            if getattr(cfg, 'max_train_batches', None) and step >= cfg.max_train_batches:
                break

        train_metrics = {"loss": tot_loss/max(tot_sent,1), "rougeL": 0.0}
        logger.log({"epoch": epoch, "split": "train", **train_metrics})
        print(f"[Epoch {epoch}] train: {train_metrics}")

        # ---- Validate ----
        val_metrics = evaluate(encoder, head, val_loader, device, cfg)
        logger.log({"epoch": epoch, "split": "val", **val_metrics})
        print(f"[Epoch {epoch}]   val: {val_metrics}")

        # Save rolling checkpoints
        if not cfg.freeze_encoder:
            torch.save(encoder.state_dict(), os.path.join(cfg.checkpoints_dir, cfg.save_encoder_as))
        torch.save(head.state_dict(), os.path.join(cfg.checkpoints_dir, cfg.save_head_as))

        # Save best
        metric_now = -val_metrics["loss"] if cfg.save_best_by == "loss" else val_metrics.get("rougeL", 0.0)
        improved = (val_metrics["loss"] < best_metric) if cfg.save_best_by == "loss" else (metric_now > best_metric)
        if improved:
            best_metric = metric_now
            if not cfg.freeze_encoder:
                torch.save(encoder.state_dict(), os.path.join(cfg.checkpoints_dir, "best_" + cfg.save_encoder_as))
            torch.save(head.state_dict(), os.path.join(cfg.checkpoints_dir, "best_" + cfg.save_head_as))
            print(f"[Best] Updated best based on {cfg.save_best_by}={val_metrics.get(cfg.save_best_by, 0.0):.4f}")

    # ---- Test ----
    test_metrics = evaluate(encoder, head, test_loader, device, cfg)
    logger.log({"epoch": cfg.epochs, "split":"test", **test_metrics})
    print(f"[Test] {test_metrics}")

def main(cfg: Optional[SumExtConfig] = None):
    if cfg is None:
        cfg = SumExtConfig()
    train(cfg)

if __name__ == "__main__":
    main()
