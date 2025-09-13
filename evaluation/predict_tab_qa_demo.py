import os, argparse, random, re
import torch, pandas as pd
from typing import Tuple

# --- robust imports (works with flat or package layout) ---
try:
    from config_tab_qa import TabQAConfig
except ImportError:
    from utils.config_tab_qa import TabQAConfig  # fallback if you keep it under utils/

try:
    from train_tab_qa import load_shared_encoder, encode_to_sequence
except ImportError:
    # If train_tab_qa.py lives under a src/ or similar, tweak your PYTHONPATH or this import.
    from src.train_tab_qa import load_shared_encoder, encode_to_sequence  # last-resort fallback

try:
    from tabqa_span_head import TabQASpanHead
except ImportError:
    from heads.tabqa_span_head import TabQASpanHead

try:
    from tabqa_synth import _serialize_row_for_tableqa, _bytes_to_ids
except ImportError:
    from data_modules.tabqa_synth import _serialize_row_for_tableqa, _bytes_to_ids


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    from collections import Counter
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0: return 0.0
    prec, rec = overlap/len(p), overlap/len(g)
    return 2*prec*rec/(prec+rec)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="data/adult.csv")
    ap.add_argument("--shared_ckpt", default="/users/acp24kp/finetuning/multiModalML/checkpoints/V1/model.pth")
    ap.add_argument("--head_ckpt",   default="/users/acp24kp/finetuning/multiModalML/checkpoints/tab_qa_adult/tab_qa_span_head.pth")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--show",  type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = TabQAConfig()
    cfg.init_checkpoint = args.shared_ckpt

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # --- load encoder & infer D ---
    enc = load_shared_encoder(cfg.init_checkpoint, device, cfg).eval()
    fake = torch.zeros(1, getattr(cfg, "src_max_len"), dtype=torch.long).to(device)
    fake_m = torch.zeros_like(fake).to(device)
    D = encode_to_sequence(enc, fake, fake_m, device).size(-1)

    # --- load head ---
    head = TabQASpanHead(hidden_dim=D).to(device).eval()
    head.load_state_dict(torch.load(args.head_ckpt, map_location=device))

    # --- data: synth questions from adult.csv ---
    df = pd.read_csv(args.csv_path)
    if args.limit: df = df.sample(args.limit, random_state=0)
    cols = list(df.columns)
    rng = random.Random(0)

    em_sum = 0.0
    f1_sum = 0.0
    shown = 0
    n = 0

    for _, r in df.iterrows():
        row = r.to_dict()
        if len(cols) < 3:
            continue

        target = rng.choice(cols)
        sel = rng.sample([c for c in cols if c != target], k=2)
        q = f"What is the {target} for the row with {sel[0]}={row[sel[0]]} and {sel[1]}={row[sel[1]]}?"
        gold = str(row[target]).strip()

        # serializer kwarg compatibility: prefer 'decompose_date', fallback to 'include_date_decompose'
        prompt_prefix = f"Q: {q}\n"
        try:
            row_str, _ = _serialize_row_for_tableqa(
                row, add_schema=False, decompose_date=getattr(cfg, "include_date_decompose", False)
            )
        except TypeError:
            row_str, _ = _serialize_row_for_tableqa(
                row, add_schema=False, include_date_decompose=getattr(cfg, "include_date_decompose", False)
            )

        s = f"{prompt_prefix}{row_str}"
        b = s.encode("utf-8", "ignore")
        ids, attn = _bytes_to_ids(b, cfg.src_max_len, cfg.pad_token, cfg.remap_255_to)

        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        m = torch.tensor(attn, dtype=torch.long).unsqueeze(0).to(device)

        seq = encode_to_sequence(enc, x, m, device)
        start_logits, end_logits = head(seq, m)  # (B,T),(B,T)

        # forbid spans from the question prefix (measured in BYTES; byte-level tokens => aligned)
        qlen = len(prompt_prefix.encode("utf-8", "ignore"))
        start_logits = start_logits.clone()
        end_logits   = end_logits.clone()
        start_logits[0, :qlen] = -1e9
        end_logits[0,   :qlen] = -1e9

        start = int(torch.argmax(start_logits, dim=-1).item())
        end   = int(torch.argmax(end_logits,   dim=-1).item())
        if end < start:
            end = start

        real_len = int(m[0].sum().item())
        # reverse the 255-remap: map 'remap_255_to' back to 255
        toks = x[0][:real_len].tolist()
        payload = [255 if t == cfg.remap_255_to else int(t) for t in toks]
        pred = bytes(payload)[start:end+1].decode("utf-8", "ignore").strip()

        em  = 1.0 if _normalize(pred) == _normalize(gold) else 0.0
        f1  = _f1(pred, gold)
        em_sum += em
        f1_sum += f1
        n += 1

        if shown < args.show:
            print(f"--- Example {shown+1} ---")
            print("Q:", q)
            print("Pred:", pred)
            print("Gold:", gold)
            shown += 1

    print("\nSamples evaluated:", n)
    print(f"Exact Match: {em_sum / max(1,n):.4f}")
    print(f"Token F1:     {f1_sum / max(1,n):.4f}")

if __name__ == "__main__":
    main()
