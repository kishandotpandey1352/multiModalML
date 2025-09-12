# predict_tab_bin_demo.py
import argparse
import os
import torch
import torch.nn.functional as F

from utils.config_tab_bin import TabBinConfig
from train_tab_qa import load_shared_encoder, encode_to_sequence
from heads.tab_binary_head import TabBinaryHead
from data_modules.tab_hf import make_loader


@torch.no_grad()
def masked_mean_pool(seq: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """
    seq:  [B, T, D]
    attn: [B, T] with 1 for real tokens, 0 for pads
    returns: [B, D]
    """
    mask = attn.float()
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # avoid div by zero
    pooled = (seq * mask.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


def infer_hidden_dim(encoder, cfg, device) -> int:
    """
    Build a tiny fake batch and run one forward pass to infer D.
    IMPORTANT: mask must contain at least one valid token (not all zeros),
    otherwise PyTorch's nested tensor path can error out.
    """
    fake_ids = torch.zeros(1, cfg.src_max_len, dtype=torch.long, device=device)
    fake_mask = torch.ones(1, cfg.src_max_len, dtype=torch.long, device=device)  # <- fix
    # If you prefer minimal valid tokens:
    # fake_mask.zero_(); fake_mask[:, 0] = 1

    seq = encode_to_sequence(encoder, fake_ids, fake_mask, device)
    return int(seq.size(-1))


def load_head(inferred_D: int, ckpt_path: str, device: str):
    head = TabBinaryHead(hidden_dim=inferred_D).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(sd)
    head.eval()

    # small debug: print first Linear in_features
    for m in head.modules():
        if isinstance(m, torch.nn.Linear):
            print(f"[head] first Linear in_features={m.in_features}, out_features={m.out_features}")
            break
    return head


def pick_head_ckpt(user_path: str | None) -> str:
    if user_path and os.path.isfile(user_path):
        return user_path

    # common fallbacks
    candidates = [
        "checkpoints/tab_bin/tab_bin_head.pth",
        "checkpoints/tab_bin/best_head.pth",
        "checkpoints/tab_bin/last_head.pth",
    ]
    abspaths = [os.path.abspath(p) for p in candidates]
    for p in abspaths:
        if os.path.isfile(p):
            return p

    tried = "\n  ".join(abspaths if not user_path else [user_path] + abspaths)
    raise FileNotFoundError(f"[error] No head checkpoint found. Tried:\n  {tried}")


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # ---- Config & encoder ----
    cfg = TabBinConfig()
    encoder = load_shared_encoder(cfg.init_checkpoint, device, cfg)
    encoder.eval()

    # ---- hidden dim ----
    D = infer_hidden_dim(encoder, cfg, device)

    # ---- load head ----
    ckpt_path = pick_head_ckpt(args.head_ckpt)
    print(f"[head] loaded: {ckpt_path}")
    head = load_head(D, ckpt_path, device)

    # ---- data loader (use test split size) ----
    bs = args.batch_size or getattr(cfg, "val_batch_size", 64)
    # Use the same split you evaluated with; here, we just use the validation/test path:
    loader = make_loader(cfg, getattr(cfg, "split_val", "test"), bs)

    # ---- evaluate a few batches ----
    total = 0
    correct = 0
    all_probs = []
    all_labels = []

    for i, batch in enumerate(loader):
        if args.n_batches is not None and i >= args.n_batches:
            break

        x = batch["input_ids"].to(device, non_blocking=True)
        m = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True).long()

        seq = encode_to_sequence(encoder, x, m, device)     # [B, T, D]
        if i == 0:
            print(f"[shapes] seq={tuple(seq.shape)}  inferred_hidden_dim={D}")

        pooled = masked_mean_pool(seq, m)                   # [B, D]
        if i == 0:
            print(f"[debug] pooled shape={tuple(pooled.shape)}")

        logit = head(pooled).squeeze(-1)                    # [B]
        prob = torch.sigmoid(logit)                         # [B]
        pred = (prob >= args.threshold).long()

        all_probs.extend(prob.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

        total += y.numel()
        correct += (pred == y).sum().item()

    if total == 0:
        print("[warn] No samples were evaluated (empty dataloader?).")
        return

    acc = correct / total
    # Show a tiny snapshot
    view_n = min(8, len(all_probs))
    probs_snip = [round(p, 3) for p in all_probs[:view_n]]
    golds_snip = all_labels[:view_n]
    preds_snip = [1 if p >= args.threshold else 0 for p in all_probs[:view_n]]

    print(f"[result] batches={args.n_batches if args.n_batches is not None else 'all'}  "
          f"threshold={args.threshold:.2f}  acc={acc:.4f}  total={total}")
    print(f"[sample] probs: {probs_snip}")
    print(f"[sample] golds: {golds_snip}")
    print(f"[sample] preds: {preds_snip}")
    print("[done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict on Adult tabular binary head (demo).")
    ap.add_argument("--head_ckpt", type=str, default=None,
                    help="Path to tab-binary head checkpoint. If omitted, common defaults are tried.")
    ap.add_argument("--n_batches", type=int, default=None,
                    help="How many batches to evaluate (default: all).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Sigmoid threshold for positive class (default: 0.5).")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Override eval batch size (default: cfg.val_batch_size or 64).")
    args = ap.parse_args()
    main(args)
