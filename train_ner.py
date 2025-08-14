
import os, random, math
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config_ner import NERTrainingConfig
from utils.logger import CSVLogger
from data_modules.wnut_hf import WNUT17HFBytes, wnut_collate_fn
from heads.byte_token_head import ByteTokenClassifierHead
from encoder.byte_encoder import ByteEncoder  # your shared encoder

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

def build_model(cfg: NERTrainingConfig, num_tags: int):
    enc_cfg = {
        "embed_dim": cfg.embed_dim,
        "max_length": cfg.max_len,
        "nhead": cfg.nhead,
        "dim_feedforward": cfg.dim_feedforward,
        "dropout": cfg.dropout,
        "num_layers": cfg.num_layers,
    }
    encoder = ByteEncoder(enc_cfg)
    head = ByteTokenClassifierHead(d_model=cfg.embed_dim, num_tags=num_tags, dropout=cfg.dropout)
    return encoder, head

def encode_to_sequence(encoder, x, attn=None):
    
    if attn is not None and attn.device != x.device:
        attn = attn.to(x.device, non_blocking=True)
    try:
        out = encoder(x, attention_mask=attn, return_hidden=True)
        return out if not isinstance(out, tuple) else out[0]
    except TypeError:
        pass
    if hasattr(encoder, "forward_hidden"):
        return encoder.forward_hidden(x, attention_mask=attn)
    out = encoder(x)
    if isinstance(out, torch.Tensor) and out.dim() == 3:
        return out
    raise RuntimeError("Encoder must return (B,T,D)")

def make_loaders(cfg: NERTrainingConfig, _label_list_unused: List[str]):
    # Build train ds WITHOUT forcing label_list; let it introspect features
    train_ds = WNUT17HFBytes(
        split=cfg.split_train, tag_field="ner_tags",
        max_len=cfg.max_len, pad_token=cfg.pad_token,
        bos_token=cfg.bos_token, eos_token=cfg.eos_token,
        add_bos=cfg.add_bos, add_eos=cfg.add_eos,
        ignore_index=cfg.ignore_index, use_hf_streaming=cfg.use_hf_streaming,
        dataset_name=cfg.dataset_name,
        dataset_config=getattr(cfg, "dataset_config", None),
    )
    # Reuse the discovered label list for val/test
    val_ds = WNUT17HFBytes(
        split=cfg.split_val, tag_field="ner_tags",
        max_len=cfg.max_len, pad_token=cfg.pad_token,
        bos_token=cfg.bos_token, eos_token=cfg.eos_token,
        add_bos=cfg.add_bos, add_eos=cfg.add_eos,
        ignore_index=cfg.ignore_index, use_hf_streaming=cfg.use_hf_streaming,
        label_list=train_ds.label_list,
        dataset_name=cfg.dataset_name,
        dataset_config=getattr(cfg, "dataset_config", None),
    )
    test_ds = WNUT17HFBytes(
        split=cfg.split_test, tag_field="ner_tags",
        max_len=cfg.max_len, pad_token=cfg.pad_token,
        bos_token=cfg.bos_token, eos_token=cfg.eos_token,
        add_bos=cfg.add_bos, add_eos=cfg.add_eos,
        ignore_index=cfg.ignore_index, use_hf_streaming=cfg.use_hf_streaming,
        label_list=train_ds.label_list,
        dataset_name=cfg.dataset_name,
        dataset_config=getattr(cfg, "dataset_config", None),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train_batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory if hasattr(cfg, "pin_memory") else True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory if hasattr(cfg, "pin_memory") else True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory if hasattr(cfg, "pin_memory") else True,
    )
    return train_loader, val_loader, test_loader, train_ds.label_list


def bio_entities_from_ids(id_seq: List[int], id2label: Dict[int,str]) -> List[tuple]:
    """Convert BIO tag id sequence to list of (type, start, end) spans [inclusive, exclusive]."""
    spans = []
    current_type = None
    start = None
    for i, tid in enumerate(id_seq):
        tag = id2label.get(tid, "O")
        if tag == "O":
            if current_type is not None:
                spans.append((current_type, start, i))
                current_type, start = None, None
            continue
        prefix, etype = tag.split("-", 1)
        if prefix == "B":
            if current_type is not None:
                spans.append((current_type, start, i))
            current_type, start = etype, i
        elif prefix == "I":
            if current_type != etype:
                # broken I without preceding B -> start new
                if current_type is not None:
                    spans.append((current_type, start, i))
                current_type, start = etype, i
    if current_type is not None:
        spans.append((current_type, start, len(id_seq)))
    return spans

def f1_score(preds: List[int], golds: List[int], ignore_index: int) -> float:
    # token-level micro F1 on supervised positions
    from sklearn.metrics import f1_score as sk_f1
    mask = [g != ignore_index for g in golds]
    if not any(mask):
        return 0.0
    p = [pi for pi, m in zip(preds, mask) if m]
    g = [gi for gi, m in zip(golds, mask) if m]
    return float(sk_f1(g, p, average="micro"))

def entity_f1_batch(pred_ids: torch.Tensor, gold_ids: torch.Tensor, id2label: Dict[int,str], ignore_index: int) -> float:
    # pred_ids, gold_ids: (B, T)
    tp = fp = fn = 0
    B, T = pred_ids.shape
    for b in range(B):
        # Consider only positions with supervision (ignore_index excluded)
        gold_seq = [int(x) for x in gold_ids[b].tolist() if x != ignore_index]
        pred_seq = [int(x) for x, g in zip(pred_ids[b].tolist(), gold_ids[b].tolist()) if g != ignore_index]
        gold_spans = set(bio_entities_from_ids(gold_seq, id2label))
        pred_spans = set(bio_entities_from_ids(pred_seq, id2label))
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(f1)

def evaluate(encoder, head, loader, device, ignore_index: int, id2label: Dict[int,str]) -> Dict[str,float]:
    encoder.eval(); head.eval()
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    total = count = 0
    loss_sum = 0.0
    tok_f1_sum = 0.0
    ent_f1_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            # attention mask is currently unused; your encoder may accept it
            hidden =  encode_to_sequence(encoder, x, batch.get("attention_mask"))    # (B, T, D)
            assert hidden.dim() == 3, hidden.shape
            logits = head(hidden)                      # (B, T, C)
            loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
            loss_sum += loss.item() * x.size(0)
            # predictions
            pred = logits.argmax(-1)                   # (B, T)
            tok_f1_sum += f1_score(pred.view(-1).tolist(), y.view(-1).tolist(), ignore_index)
            ent_f1_sum += entity_f1_batch(pred, y, id2label, ignore_index)
            total += x.size(0); count += 1
    return {
        "loss": loss_sum/total,
        "token_f1": tok_f1_sum/max(count,1),
        "entity_f1": ent_f1_sum/max(count,1),
    }

def train(cfg: NERTrainingConfig):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build data
    train_loader, val_loader, test_loader, label_list = make_loaders(cfg, cfg.label_list)
    num_tags = len(label_list)
    id2label = {i:l for i,l in enumerate(label_list)}

    # Build model
    encoder, head = build_model(cfg, num_tags)
    load_pretrained_encoder(encoder, cfg.init_checkpoint)
    encoder.to(device); head.to(device)
    
    # --- SMOKE TEST (one batch) ---
    b = next(iter(train_loader))
    print("sample shapes:", "input_ids", tuple(b["input_ids"].shape), "labels", tuple(b["labels"].shape))
    with torch.no_grad():
        x = b["input_ids"].to(device)
        am = b.get("attention_mask")
        if am is not None:
            am = am.to(device, non_blocking=True)
        h = encode_to_sequence(encoder, x, am)
        print("hidden shape:", tuple(h.shape))
    # ------------------------------

    # Freeze if requested
    if cfg.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

    # Optimizer
    params = [{"params": head.parameters(), "lr": cfg.lr_head}]
    if not cfg.freeze_encoder:
        params.append({"params": encoder.parameters(), "lr": cfg.lr_encoder})
    optim = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device == "cuda")

    # Logger
    logger = CSVLogger(cfg.csv_log_path, fieldnames=["epoch","split","loss","token_f1","entity_f1"])

    best_metric = -1e9 if cfg.save_best_by != "loss" else 1e9
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    enc_path = os.path.join(cfg.checkpoints_dir, cfg.save_encoder_as)
    head_path = os.path.join(cfg.checkpoints_dir, cfg.save_head_as)

    for epoch in range(1, cfg.epochs+1):
        # === Train epoch ===
        if not cfg.freeze_encoder:
            encoder.train()
        else:
            encoder.eval()
        head.train()

        total = 0
        loss_sum = 0.0
        tok_f1_sum = 0.0
        ent_f1_sum = 0.0
        count = 0

        for batch in train_loader:
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device == "cuda"):
                hidden =  encode_to_sequence(encoder, x, batch.get("attention_mask"))      # (B,T,D)
                assert hidden.dim() == 3, hidden.shape
                logits = head(hidden)                        # (B,T,C)
                loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
                if not cfg.freeze_encoder:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()

            # metrics (quick, token-level micro F1 per batch)
            pred = logits.argmax(-1)
            tok_f1_sum += f1_score(pred.view(-1).tolist(), y.view(-1).tolist(), cfg.ignore_index)
            ent_f1_sum += entity_f1_batch(pred, y, id2label, cfg.ignore_index)
            loss_sum += loss.item() * x.size(0)
            total += x.size(0); count += 1

        train_metrics = {
            "loss": loss_sum/total,
            "token_f1": tok_f1_sum/max(count,1),
            "entity_f1": ent_f1_sum/max(count,1),
        }
        logger.log({"epoch": epoch, "split":"train", **train_metrics})
        print(f"[Epoch {epoch}] train: {train_metrics}")

        # === Validate ===
        val_metrics = evaluate(encoder, head, val_loader, device, cfg.ignore_index, id2label)
        logger.log({"epoch": epoch, "split":"val", **val_metrics})
        print(f"[Epoch {epoch}]   val: {val_metrics}")

        # Save rolling checkpoints
        if not cfg.freeze_encoder:
            torch.save(encoder.state_dict(), enc_path)
        torch.save(head.state_dict(), head_path)

        # Save best
        current = -val_metrics["loss"] if cfg.save_best_by == "loss" else val_metrics.get("entity_f1", 0.0)
        if (cfg.save_best_by == "loss" and val_metrics["loss"] < best_metric) or \
           (cfg.save_best_by != "loss" and current > best_metric):
            best_metric = current
            if not cfg.freeze_encoder:
                torch.save(encoder.state_dict(), os.path.join(cfg.checkpoints_dir, "best_" + cfg.save_encoder_as))
            torch.save(head.state_dict(), os.path.join(cfg.checkpoints_dir, "best_" + cfg.save_head_as))
            print(f"[Best] Updated best model based on {cfg.save_best_by}={val_metrics.get(cfg.save_best_by, val_metrics['entity_f1']):.4f}")

    # === Test ===
    test_metrics = evaluate(encoder, head, test_loader, device, cfg.ignore_index, id2label)
    logger.log({"epoch": cfg.epochs, "split":"test", **test_metrics})
    print(f"[Test] {test_metrics}")

def main(cfg: NERTrainingConfig | None = None):
    if cfg is None:
        cfg = NERTrainingConfig()
    train(cfg)

if __name__ == "__main__":
    main()
