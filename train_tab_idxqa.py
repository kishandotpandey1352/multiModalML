# train_tab_idxqa.py
import os, csv, math, time, random
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config_tab_idxqa import TabIdxQAConfig
from data_modules.tab_idxqa import make_loader
from heads.tabqa_span_head import TabQASpanHead

# Reuse your existing helpers if available
from train_tab_qa import load_shared_encoder, encode_to_sequence

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def mask_logits(logits, attn, win_start=None, win_end=None, use_window=True):
    # logits: [B,T], attn: [B,T] (1 real, 0 pad)
    mask = (attn == 0)
    if use_window and (win_start is not None) and (win_end is not None):
        B, T = logits.size()
        ar = torch.arange(T, device=logits.device).unsqueeze(0).expand(B, T)
        mask = mask | (ar < win_start.unsqueeze(1)) | (ar > win_end.unsqueeze(1))
    return logits.masked_fill(mask, -1e9)

def span_loss(start_logits, end_logits, start_idx, end_idx):
    return F.cross_entropy(start_logits, start_idx) + F.cross_entropy(end_logits, end_idx)

def char_f1(pred: str, gold: str) -> float:
    # simple char-level F1
    ps = list(pred); gs = list(gold)
    common = 0
    g_count = {}
    for ch in gs:
        g_count[ch] = g_count.get(ch, 0) + 1
    for ch in ps:
        if g_count.get(ch, 0) > 0:
            common += 1
            g_count[ch] -= 1
    prec = common / max(1, len(ps))
    rec  = common / max(1, len(gs))
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)

def decode_answer(bytes_seq: bytes, start: int, end: int) -> str:
    end = max(start, end)
    byte_slice = bytes_seq[start:end+1]
    return byte_slice.decode("utf-8", "ignore").strip()

def maybe_unfreeze_last_n(encoder: nn.Module, n_layers: int) -> int:
    """
    Try to unfreeze last n transformer blocks by name.
    Adjust this to match your encoder's module names if needed.
    """
    if n_layers <= 0:
        for p in encoder.parameters(): p.requires_grad = False
        return 0
    # naive fallback: unfreeze a suffix of parameters
    total = sum(1 for _ in encoder.parameters())
    cut = max(0, total - n_layers*12)  # heuristic; tweak if necessary
    cnt = 0
    for i, p in enumerate(encoder.parameters()):
        p.requires_grad = i >= cut
        if p.requires_grad: cnt += 1
    return cnt

def main(cfg: TabIdxQAConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(Path(cfg.csv_log_path).parent, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    print(f"[cfg] {cfg}")
    print(f"[device] selected={device}  cuda.is_available={torch.cuda.is_available()}")

    # Loaders
    train_loader = make_loader(cfg, "train", cfg.train_batch_size)
    val_loader   = make_loader(cfg, "val",   cfg.val_batch_size)

    # Build first batch to infer hidden dim
    first = next(iter(train_loader))
    print("[train] first batch:",
          {k: (tuple(v.shape) if torch.is_tensor(v) else f"list(len={len(v)})")
           for k, v in first.items()})

    # Load encoder
    encoder = load_shared_encoder(cfg.init_checkpoint, device, cfg)
    unfrozen = maybe_unfreeze_last_n(encoder, cfg.unfreeze_n_layers)
    print(f"[encoder] params_unfrozen={unfrozen}")

    # Infer hidden size
    fake_ids  = torch.zeros(1, cfg.src_max_len, dtype=torch.long, device=device)
    fake_mask = torch.zeros(1, cfg.src_max_len, dtype=torch.long, device=device)
    # ensure at least one valid token so nested-tensor path doesn't see empty seq
    fake_mask[:, 0] = 1
    
    with torch.no_grad():
        D = encode_to_sequence(encoder, fake_ids, fake_mask, device).size(-1)
    print(f"[model] hidden_dim D={D}")

    head = TabQASpanHead(hidden_dim=D).to(device)

    # Optim + scheduler (warmup → cosine)
    params = list(head.parameters()) + [p for p in encoder.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr_head, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * cfg.train_steps_per_epoch
    warmup = int(0.1 * total_steps)
    def lr_lambda(step):
        if step < warmup: return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device == "cuda")

    # CSV logger
    fieldnames = ["epoch", "train_loss", "val_loss", "em", "f1", "lr"]
    csv_f = open(cfg.csv_log_path, "w", newline=""); writer = csv.DictWriter(csv_f, fieldnames=fieldnames); writer.writeheader()

    best_em = -1.0
    global_step = 0

    print("[setup] starting training...")
    for epoch in range(1, cfg.epochs + 1):
        encoder.train(); head.train()
        t0 = time.time(); running = 0.0

        it = iter(train_loader)
        for step in range(1, cfg.train_steps_per_epoch + 1):
            batch = next(it)
            x = batch["input_ids"].to(device, non_blocking=True)
            m = batch["attention_mask"].to(device, non_blocking=True)
            s = batch["start_idx"].to(device, non_blocking=True)
            e = batch["end_idx"].to(device, non_blocking=True)
            ws = batch["win_start"].to(device, non_blocking=True)
            we = batch["win_end"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device == "cuda"):
                seq = encode_to_sequence(encoder, x, m, device)  # [B,T,D]
                start_logits, end_logits = head(seq, m)          # [B,T],[B,T]
                start = mask_logits(start_logits, m, ws, we, use_window=cfg.windowed_loss)
                end   = mask_logits(end_logits,   m, ws, we, use_window=cfg.windowed_loss)
                loss = span_loss(start, end, s, e)

            scaler.scale(loss).backward()
            if cfg.clip_grad_norm is not None:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(params, cfg.clip_grad_norm)
            scaler.step(optim); scaler.update()
            scheduler.step()

            running += loss.item()
            global_step += 1
            if step % cfg.log_every_n == 0:
                dt = time.time() - t0
                lr = optim.param_groups[0]["lr"]
                print(f"[train] step {step} loss={loss.item():.4f} lr={lr:.2e} dt={dt:.2f}s")
                t0 = time.time()

        train_loss = running / cfg.train_steps_per_epoch

        # ---- Validation ----
        encoder.eval(); head.eval()
        v_running = 0.0; em_sum = 0.0; f1_sum = 0.0; n = 0
        with torch.no_grad():
            itv = iter(val_loader)
            for _ in range(cfg.val_steps):
                batch = next(itv)
                x = batch["input_ids"].to(device, non_blocking=True)
                m = batch["attention_mask"].to(device, non_blocking=True)
                s = batch["start_idx"].to(device, non_blocking=True)
                e = batch["end_idx"].to(device, non_blocking=True)

                seq = encode_to_sequence(encoder, x, m, device)
                start_logits, end_logits = head(seq, m)
                start = mask_logits(start_logits, m, batch["win_start"].to(device), batch["win_end"].to(device),
                                    use_window=cfg.windowed_loss)
                end   = mask_logits(end_logits,   m, batch["win_start"].to(device), batch["win_end"].to(device),
                                    use_window=cfg.windowed_loss)
                loss = span_loss(start, end, s, e)
                v_running += loss.item()

                # decode predictions
                start_idx = start.argmax(dim=-1)  # [B]
                end_idx   = end.argmax(dim=-1)
                for b in range(x.size(0)):
                    valid = int(m[b].sum().item())
                    # reconstruct bytes and decode slice
                    raw_bytes = bytes([(cfg.remap_255_to if t == cfg.pad_token else t) for t in x[b, :valid].tolist()])
                    pred = decode_answer(raw_bytes, int(start_idx[b]), int(end_idx[b]))
                    gold = batch["answer_text"][b] if isinstance(batch["answer_text"], list) else str(batch["answer_text"][b])
                    em = 1.0 if pred == gold else 0.0
                    f1 = char_f1(pred, gold)
                    em_sum += em; f1_sum += f1; n += 1

        val_loss = v_running / max(1, cfg.val_steps)
        em = em_sum / max(1, n)
        f1 = f1_sum / max(1, n)
        writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "em": em, "f1": f1,
                         "lr": optim.param_groups[0]["lr"]}); csv_f.flush()
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  EM={em:.3f}  F1={f1:.3f}")

        # Checkpoint
        best_path = os.path.join(cfg.save_dir, cfg.save_head_as)
        last_path = os.path.join(cfg.save_dir, "last_head.pth")
        if em > best_em:
            best_em = em
            torch.save(head.state_dict(), best_path)
            print(f"[ckpt] Saved best head → {best_path}")
        torch.save(head.state_dict(), last_path)
        print(f"[ckpt] Saved last head → {last_path}")

if __name__ == "__main__":
    cfg = TabIdxQAConfig()
    main(cfg)
