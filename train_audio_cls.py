
# train_audio_cls.py — safe, memory-capped trainer for audio classification (byte encoder)
# - No key-padding mask into Transformer (avoids nested-tensor path)
# - Encoder micro-batching to cap attention peak memory
# - Optional eval skipping + step caps for smoke tests
# - Optional "freeze_first_batch" to avoid streaming iterator pressure
# - AMP + TF32 enabled for speed/efficiency on Ampere+
# - Optional SAFE MODE to bypass attention: set env MM_SAFE_NO_ATTN=1

import os, csv, random, gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_modules.audio_hf import make_loader
from heads.img_classifier_head import ImageClassifierHead  # generic seq→class head

# Prefer flash/mem-efficient SDPA kernels; keep math fallback enabled
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash(True)
    sdp_kernel.enable_mem_efficient(True)
    sdp_kernel.enable_math(True)
except Exception:
    pass

# Allow TF32 on Ampere/Hopper (safe and helpful)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_shared_encoder(init_checkpoint: str, device: str, cfg=None):
    import importlib.util, dataclasses
    from encoder.byte_encoder import ByteEncoder

    def _obj_to_dict(obj):
        if isinstance(obj, dict): return obj
        if dataclasses.is_dataclass(obj): return dataclasses.asdict(obj)
        d = {}
        for k in dir(obj):
            if k.startswith("_"): continue
            v = getattr(obj, k)
            if isinstance(v, (int,float,bool,str,tuple,list,dict)) or v is None: d[k]=v
        return d

    def _normalize(d: dict) -> dict:
        out = dict(d)
        # unify common alias keys from configs
        if "max_length" not in out and "max_len" in out: out["max_length"]=out["max_len"]
        if "dim_feedforward" not in out and "dim_feed_forward" in out: out["dim_feedforward"]=out["dim_feed_forward"]
        # sensible defaults
        out.setdefault("embed_dim", 512)
        out.setdefault("max_length", 1024)
        out.setdefault("nhead", 8)
        out.setdefault("num_layers", 6)
        out.setdefault("dim_feedforward", 2048)
        out.setdefault("dropout", 0.1)
        return out

    # Load encoder config module/class by path
    path = os.path.abspath(getattr(cfg, "encoder_config_path", "utils/config.py"))
    spec = importlib.util.spec_from_file_location("enc_cfg_mod", path)
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(mod)
    cls = getattr(mod, getattr(cfg, "encoder_config_class", None) or "TrainingConfig", None) \
          or getattr(mod, "TrainingConfig", None) or getattr(mod, "Config", None)

    cfg_map = _normalize(_obj_to_dict(cls() if callable(cls) else cls))

    enc = ByteEncoder(cfg_map)
    sd = torch.load(init_checkpoint, map_location="cpu")
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded encoder (strict=False). missing={len(missing)} unexpected={len(unexpected)}")
    enc.eval()

    # Optional: bypass attention entirely (for diagnosis)
    force_safe = bool(getattr(cfg, "force_safe_encoder", False) or os.getenv("MM_SAFE_NO_ATTN") == "1")
    if force_safe:
        class _Noop(nn.Module):
            def forward(self, x, src_key_padding_mask=None):
                return x
        enc.transformer_encoder = _Noop()
        print("[INFO] SAFE MODE: transformer encoder bypassed.")

    return enc.to(device)


@torch.no_grad()
def encode_to_sequence(encoder, x, attn, device):
    max_len = getattr(encoder, "max_length", x.size(1))
    if x.size(1) > max_len:
        x = x[:, :max_len]
        if attn is not None:
            attn = attn[:, :max_len]
        if not hasattr(encode_to_sequence, "_warned"):
            print(f"[WARN] Truncating sequence to encoder.max_length={max_len}.")
            encode_to_sequence._warned = True

    # NEW: pass attention_mask instead of None
    out = encoder(
        x.to(device, non_blocking=True),
        attention_mask=attn.to(device, non_blocking=True) if attn is not None else None,
        return_hidden=True
    )
    return out if not isinstance(out, tuple) else out[0]


def topk_acc(logits, target, ks=(1,)):
    with torch.no_grad():
        maxk = max(ks)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in ks:
            res.append(correct[:k].reshape(-1).float().sum(0).item() / target.size(0))
        return res


def train_one_epoch(cfg, encoder, head, loader, device, scaler, optim, epoch):
    encoder.requires_grad_(False); head.train()
    ce = nn.CrossEntropyLoss()
    tot, correct, n = 0.0, 0, 0
    max_steps = int(getattr(cfg, "max_train_steps", 0) or 0)

    for step, batch in enumerate(loader, 1):
        x = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        with torch.cuda.amp.autocast(enabled=getattr(cfg, "amp", True) and device == 'cuda'):
            if step == 1 and epoch == 1:
                print("[DBG] before encode", tuple(x.shape), flush=True)
            hidden = encode_to_sequence(encoder, x, attn, device)
            if step == 1 and epoch == 1:
                print("[DBG] after encode", tuple(hidden.shape), flush=True)
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
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)

        # Aggressive cleanup under tight cgroup limits
        del hidden, logits, loss; gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()

        if step % cfg.log_every_n == 0:
            print(f"[Epoch {epoch}] step {step} loss={tot/max(n,1):.4f} acc={correct/max(n,1):.4f}")
        if max_steps and step >= max_steps:
            break

    return tot / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(cfg, encoder, head, loader, device):
    encoder.eval(); head.eval()
    ce = nn.CrossEntropyLoss()
    tot, correct1, correct5, n = 0.0, 0, 0, 0
    max_batches = int(getattr(cfg, "max_val_batches", 0) or 0)

    for i, batch in enumerate(loader, 1):
        x = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        with torch.cuda.amp.autocast(enabled=getattr(cfg, "amp", True) and device == 'cuda'):
            hidden = encode_to_sequence(encoder, x, attn, device)
        logits = head(hidden, attn)
        loss = ce(logits, y)

        tot += loss.item() * y.size(0)
        t1, t5 = topk_acc(logits, y, ks=(1, 5))
        correct1 += int(t1 * y.size(0)); correct5 += int(t5 * y.size(0))
        n += y.size(0)

        del hidden, logits, loss; gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()
        if max_batches and i >= max_batches:
            break

    return tot / max(n, 1), correct1 / max(n, 1), correct5 / max(n, 1)


class FrozenBatchLoader:
    """
    Replays a single decoded batch 'steps' times.
    Useful for smoke tests: avoids streaming/iterator pressure.
    """
    def __init__(self, batch, steps: int = 100):
        self.batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
        self.steps = steps if steps and steps > 0 else 100
    def __iter__(self):
        for _ in range(self.steps):
            yield {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.batch.items()}
    def __len__(self):
        return self.steps


def main(cfg):
    # Keep CPU-side workers lean
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    print(">>> SAFE PATCH ACTIVE v2 <<<")
    set_seed(getattr(cfg, "seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    print("encoder cfg will use:", {
        "embed_dim": getattr(cfg, "embed_dim", "cfg file")
    })

    # Optional knobs (defaults chosen for smoke runs)
    cfg.max_train_steps = int(getattr(cfg, "max_train_steps", 0) or 0)
    cfg.max_val_batches = int(getattr(cfg, "max_val_batches", 0) or 0)
    cfg.disable_eval    = bool(getattr(cfg, "disable_eval", True))
    cfg.freeze_first_batch = bool(getattr(cfg, "freeze_first_batch", True))

    # Build train loader
    train_loader = make_loader(cfg, getattr(cfg, "split_train", "train"), cfg.batch_size)

    # Optionally: freeze and replay the first decoded batch to avoid streaming pressure
    if cfg.freeze_first_batch:
        it0 = iter(train_loader)
        first = next(it0)  # forces one decode; prints shapes
        print({k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in first.items()})
        # drop original loader to free resources
        del it0, train_loader; gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()
        steps = cfg.max_train_steps or 20
        train_loader = FrozenBatchLoader(first, steps=steps)

    # Build val loader only if evaluation is enabled
    if not cfg.disable_eval:
        val_loader = make_loader(cfg, getattr(cfg, "split_val", "test"), cfg.val_batch_size)
    else:
        val_loader = None

    # Encoder
    encoder = load_shared_encoder(cfg.init_checkpoint, device, cfg)
    print("encoder.max_length:", getattr(encoder, "max_length", None),
          "embed_dim:", getattr(encoder, "embed_dim", None))
    setattr(encoder, "_microbatch", int(getattr(cfg, "encoder_microbatch", 1)))

    # Head
    D = int(getattr(encoder, "embed_dim", 512))
    head = ImageClassifierHead(hidden_dim=D, num_classes=cfg.num_classes).to(device)

    # Optim, scaler, schedule
    optim_head = optim.AdamW(head.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    use_amp = getattr(cfg, "amp", True) and device == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    warmup_epochs = max(1, cfg.epochs // 10)
    schedule = CosineAnnealingLR(optim_head, T_max=max(1, cfg.epochs - warmup_epochs))

    def maybe_warmup(epoch):
        if epoch <= warmup_epochs:
            scale = epoch / max(1, warmup_epochs)
            for g in optim_head.param_groups:
                g["lr"] = cfg.lr_head * scale

    # Logging / checkpoints
    os.makedirs(cfg.save_dir, exist_ok=True)
    if cfg.csv_log_path:
        os.makedirs(os.path.dirname(cfg.csv_log_path), exist_ok=True)
        write_header = (not os.path.exists(cfg.csv_log_path)) or (os.path.getsize(cfg.csv_log_path) == 0)
        with open(cfg.csv_log_path, "a", newline="") as f:
            if write_header:
                csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","val_acc5"])

    best = -1.0
    for epoch in range(1, cfg.epochs + 1):
        maybe_warmup(epoch)
        tr_loss, tr_acc = train_one_epoch(cfg, encoder, head, train_loader, device, scaler, optim_head, epoch)

        if cfg.disable_eval:
            va_loss, va_acc, va_acc5 = 0.0, 0.0, 0.0
            print(f"[Epoch {epoch}] tr_acc={tr_acc:.4f} (eval disabled)")
        else:
            va_loss, va_acc, va_acc5 = evaluate(cfg, encoder, head, val_loader, device)
            print(f"[Epoch {epoch}] tr_acc={tr_acc:.4f} val@1={va_acc:.4f} val@5={va_acc5:.4f}")

        schedule.step(); gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()

        if cfg.csv_log_path:
            with open(cfg.csv_log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{va_loss:.4f}", f"{va_acc:.4f}", f"{va_acc5:.4f}"])

        if (not cfg.disable_eval) and va_acc > best:
            best = va_acc
            torch.save(head.state_dict(), os.path.join(cfg.save_dir, cfg.save_head_as))
            if cfg.save_encoder_as:
                torch.save(encoder.state_dict(), os.path.join(cfg.save_dir, cfg.save_encoder_as))
            print(f"[INFO] Saved best head (val@1={va_acc:.4f}).")


if __name__ == "__main__":
    from utils.config_audio_cls import AudioClsConfig
    main(AudioClsConfig())
