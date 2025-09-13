import os, argparse, random, re, csv
import torch, pandas as pd
from typing import List, Tuple

# --- robust imports (flat or package layout) ---
try:
    from config_tab_qa import TabQAConfig
except ImportError:
    from utils.config_tab_qa import TabQAConfig

try:
    from train_tab_qa import load_shared_encoder, encode_to_sequence
except ImportError:
    from src.train_tab_qa import load_shared_encoder, encode_to_sequence  # if you use a src/ layout

try:
    from tabqa_span_head import TabQASpanHead
except ImportError:
    from heads.tabqa_span_head import TabQASpanHead

try:
    from tabqa_synth import _serialize_row_for_tableqa, _bytes_to_ids
except ImportError:
    from data_modules.tabqa_synth import _serialize_row_for_tableqa, _bytes_to_ids


# ------------------------ text utils ------------------------

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<\s*(CAT|NUM|STR)\s*:\s*(.*?)\s*>", flags=re.IGNORECASE)
_FIELDVAL = re.compile(r"(^|[|>]\s*|\s|^)[A-Za-z0-9_.-]+\s*=\s*")
_TARGET_FROM_Q = re.compile(r"^What is the (.*?) for the row", re.IGNORECASE)

def _normalize(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())

def _clean_value(txt: str) -> str:
    if not txt:
        return txt
    t = txt.strip()

    # Drop leading "fieldname=" if present
    m = _FIELDVAL.search(t)
    if m:
        eq = t.find("=")
        if eq != -1:
            t = t[eq+1:].strip()

    # Unwrap <CAT:...>, <NUM:...>, <STR:...> tags (possibly nested)
    def _unwrap_tags(s: str) -> str:
        prev = None
        cur = s
        while prev != cur:
            prev = cur
            cur = _TAG.sub(lambda m: m.group(2), cur)
        return cur
    t = _unwrap_tags(t)

    # Trim separators/punctuation and squash spaces
    t = t.strip(" >|;,:")
    t = _WS.sub(" ", t)
    return t

def _f1(pred: str, gold: str) -> float:
    from collections import Counter
    p = _normalize(_clean_value(pred)).split()
    g = _normalize(_clean_value(gold)).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0: return 0.0
    prec, rec = overlap/len(p), overlap/len(g)
    return 2*prec*rec/(prec+rec)

def _em(pred: str, gold: str) -> float:
    return 1.0 if _normalize(_clean_value(pred)) == _normalize(_clean_value(gold)) else 0.0


# ------------------------ span search ------------------------

def best_span_from_logits(start_logits: torch.Tensor,
                          end_logits: torch.Tensor,
                          real_len: int,
                          max_answer_bytes: int = 64,
                          length_bias: float = 0.0) -> Tuple[int, int]:
    """
    Pick (start, end) maximizing start[i] + end[j] + length_bias*(j-i+1)
    with i <= j and (j - i + 1) <= max_answer_bytes.
    """
    T = int(start_logits.shape[0])
    best_score = float("-inf")
    best_i, best_j = 0, 0
    last = min(T, real_len)
    window = max(1, max_answer_bytes)
    for i in range(last):
        s = start_logits[i].item()
        if s == float("-inf"):
            continue
        j_lo = i
        j_hi = min(last - 1, i + window - 1)
        ew = end_logits[j_lo : j_hi + 1]
        if ew.numel() == 0:
            continue
        val, idx = torch.max(ew, dim=0)
        j = j_lo + int(idx.item())
        length = (j - i + 1)
        score = s + val.item() + length_bias * length
        if score > best_score:
            best_score = score
            best_i, best_j = i, j
    if best_j < best_i:
        best_j = best_i
    return best_i, best_j


# ------------------------ value-masking + snapping helpers ------------------------

_VAL_TAGS = (b"<CAT:", b"<NUM:", b"<STR:")

def _tag_prefix_len_at(b: bytes, idx: int):
    for t in _VAL_TAGS:
        if idx + len(t) <= len(b) and b[idx:idx+len(t)] == t:
            return len(t)
    return None

def build_allowed_mask_all_values(byte_seq: bytes, qlen: int, real_len: int) -> List[bool]:
    """Allow spans only inside the VALUE part of <CAT:...>, <NUM:...>, <STR:...>."""
    allowed = [False] * real_len
    i = qlen
    end = min(real_len, len(byte_seq))
    while i < end:
        j = byte_seq.find(b"<", i, end)
        if j == -1:
            break
        pref = _tag_prefix_len_at(byte_seq, j)
        if pref is None:
            i = j + 1
            continue
        val_s = j + pref
        k = byte_seq.find(b">", val_s, end)
        if k == -1:
            break
        for p in range(val_s, k):
            if p < real_len:
                allowed[p] = True
        i = k + 1
    return allowed

def build_allowed_mask_target_only(byte_seq: bytes, qlen: int, real_len: int, target_col: str) -> List[bool]:
    """Allow spans only inside the VALUE of the target column."""
    allowed = [False] * real_len
    t_bytes = (target_col + "=").encode("utf-8", "ignore")
    end = min(real_len, len(byte_seq))

    pos = byte_seq.find(t_bytes, qlen, end)
    if pos == -1:
        return allowed  # nothing allowed; caller may fall back to all-values

    # find tag start right after 'target='
    tag_start = byte_seq.find(b"<", pos + len(t_bytes), end)
    if tag_start == -1:
        return allowed
    pref = _tag_prefix_len_at(byte_seq, tag_start)
    if pref is None:
        return allowed

    val_s = tag_start + pref
    k = byte_seq.find(b">", val_s, end)
    if k == -1:
        return allowed
    for p in range(val_s, k):
        if p < real_len:
            allowed[p] = True
    return allowed

def _extract_target_from_q(q: str) -> str:
    m = _TARGET_FROM_Q.search(q)
    return m.group(1) if m else ""

def segments_from_allowed(allowed: List[int | bool]) -> List[Tuple[int,int]]:
    """Return contiguous [a,b] segments where allowed == True."""
    segs: List[Tuple[int,int]] = []
    i = 0
    n = len(allowed)
    while i < n:
        if allowed[i]:
            j = i
            while j + 1 < n and allowed[j+1]:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs

def snap_span_to_segment(start_idx: int, end_idx: int, segs: List[Tuple[int,int]]) -> Tuple[int,int]:
    """Snap (start,end) to the full contiguous allowed segment containing start (or end)."""
    for a,b in segs:
        if a <= start_idx <= b:
            return a, b
    for a,b in segs:
        if a <= end_idx <= b:
            return a, b
    return start_idx, end_idx


# ------------------------ batching helpers ------------------------

def _to_tensors(batch_bytes: List[bytes], cfg: TabQAConfig, device: str):
    ids_list, attn_list, lens = [], [], []
    for b in batch_bytes:
        ids, attn = _bytes_to_ids(b, cfg.src_max_len, cfg.pad_token, cfg.remap_255_to)
        ids_list.append(ids)
        attn_list.append(attn)
        lens.append(int(sum(attn)))
    x = torch.tensor(ids_list, dtype=torch.long, device=device)
    m = torch.tensor(attn_list, dtype=torch.long, device=device)
    return x, m, lens


# ------------------------ main eval ------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="data/adult.csv")
    ap.add_argument("--shared_ckpt", default="/users/acp24kp/finetuning/multiModalML/checkpoints/V1/model.pth")
    ap.add_argument("--head_ckpt",   default="/users/acp24kp/finetuning/multiModalML/checkpoints/tab_qa_adult/tab_qa_span_head.pth")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--show",  type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_answer_bytes", type=int, default=32, help="Cap the span length.")
    ap.add_argument("--mask_mode", choices=["none","all_values","target_only"], default="target_only",
                    help="Constrain spans to any value region or only the target field's value.")
    ap.add_argument("--add_schema", action="store_true",
                    help="Enable if training serializer included schema.")
    ap.add_argument("--snap_to_value", action="store_true", default=True,
                    help="Expand the chosen span to the full allowed value segment.")
    ap.add_argument("--length_bias", type=float, default=0.02,
                    help="Adds this * length to start+end score to prefer complete values.")
    ap.add_argument("--dump_csv", default="", help="Optional path to write per-example results")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = TabQAConfig()
    cfg.init_checkpoint = args.shared_ckpt

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # --- load encoder used during training ---
    enc = load_shared_encoder(cfg.init_checkpoint, device, cfg).eval()

    # --- infer D from head checkpoint (robust), fallback probe if needed ---
    sd_cpu = torch.load(args.head_ckpt, map_location="cpu")
    D = None
    for k in ["start_fc.weight", "classifier_start.weight", "start.weight"]:
        if isinstance(sd_cpu, dict) and k in sd_cpu:
            w = sd_cpu[k]
            if w.ndim == 2:
                if w.shape[0] == 1 and w.shape[1] >= 1: D = int(w.shape[1])
                elif w.shape[1] == 1 and w.shape[0] >= 1: D = int(w.shape[0])
            break
    if D is None:
        fake = torch.zeros(1, getattr(cfg, "src_max_len"), dtype=torch.long, device=device)
        fake_m = torch.zeros_like(fake)
        fake[0, 0] = 1
        fake_m[0, 0] = 1
        D = encode_to_sequence(enc, fake, fake_m, device).size(-1)

    # --- head ---
    head = TabQASpanHead(hidden_dim=D).to(device).eval()
    head.load_state_dict(torch.load(args.head_ckpt, map_location=device))

    # --- data: synth queries from Adult CSV ---
    df = pd.read_csv(args.csv_path)
    if args.limit: df = df.sample(args.limit, random_state=0)
    cols = list(df.columns)
    rng = random.Random(0)

    questions: List[str] = []
    golds: List[str] = []
    payloads: List[bytes] = []
    qbyte_prefix: List[int] = []
    targets: List[str] = []

    for _, r in df.iterrows():
        row = r.to_dict()
        if len(cols) < 3: continue

        target = rng.choice(cols)
        sel = rng.sample([c for c in cols if c != target], k=2)
        q = f"What is the {target} for the row with {sel[0]}={row[sel[0]]} and {sel[1]}={row[sel[1]]}?"
        gold = str(row[target]).strip()

        try:
            row_str, _ = _serialize_row_for_tableqa(
                row, add_schema=args.add_schema, decompose_date=getattr(cfg, "include_date_decompose", False)
            )
        except TypeError:
            row_str, _ = _serialize_row_for_tableqa(
                row, add_schema=args.add_schema, include_date_decompose=getattr(cfg, "include_date_decompose", False)
            )

        prefix = f"Q: {q}\n"
        s = f"{prefix}{row_str}"
        b = s.encode("utf-8", "ignore")

        questions.append(q)
        golds.append(gold)
        payloads.append(b)
        qbyte_prefix.append(len(prefix.encode("utf-8", "ignore")))
        targets.append(target)

    # --- eval loop (micro-batched) ---
    em_sum = 0.0
    f1_sum = 0.0
    n = len(payloads)
    shown = 0
    rows_out = []

    B = args.batch_size
    for off in range(0, n, B):
        j = min(n, off + B)
        batch_bytes = payloads[off:j]
        x, m, lens = _to_tensors(batch_bytes, cfg, device=device)

        # encode (B, T, D)
        seq = encode_to_sequence(enc, x, m, device)
        start_logits, end_logits = head(seq, m)  # (B,T),(B,T)

        # per-item masking: question, tail, and value-only region
        allowed_masks: List[List[bool] | None] = []  # store for snapping
        for bi in range(j - off):
            qlen = qbyte_prefix[off + bi]
            real_len = lens[bi]

            # mask question portion
            start_logits[bi, :qlen] = float("-inf")
            end_logits[bi,   :qlen] = float("-inf")
            # mask beyond real length (safety)
            if real_len < cfg.src_max_len:
                start_logits[bi, real_len:] = float("-inf")
                end_logits[bi,   real_len:] = float("-inf")

            cur_allowed = None
            # ---- values-only masking ----
            if args.mask_mode != "none":
                # reconstruct current bytes from tokens (post-remap), bounded to real_len
                toks = x[bi, :real_len].tolist()
                byte_seq = bytes([255 if t == cfg.remap_255_to else int(t) for t in toks])

                if args.mask_mode == "target_only":
                    allowed = build_allowed_mask_target_only(
                        byte_seq, qlen, real_len, targets[off + bi]
                    )
                    # if we failed to find a target value, fall back to all-values
                    if not any(allowed):
                        allowed = build_allowed_mask_all_values(byte_seq, qlen, real_len)
                else:
                    allowed = build_allowed_mask_all_values(byte_seq, qlen, real_len)

                if any(allowed):
                    disallow = torch.tensor([not z for z in allowed], dtype=torch.bool, device=device)
                    start_logits[bi, :real_len][disallow] = float("-inf")
                    end_logits[bi,   :real_len][disallow] = float("-inf")
                    cur_allowed = allowed

            allowed_masks.append(cur_allowed)

        # choose best spans
        starts: List[int] = []
        ends:   List[int] = []
        for bi in range(j - off):
            s_i, e_i = best_span_from_logits(
                start_logits[bi], end_logits[bi],
                real_len=lens[bi],
                max_answer_bytes=args.max_answer_bytes,
                length_bias=args.length_bias
            )
            # optional snap to entire allowed segment
            if args.snap_to_value and args.mask_mode != "none" and allowed_masks[bi] is not None:
                segs = segments_from_allowed(allowed_masks[bi])  # type: ignore[arg-type]
                if segs:
                    s_i, e_i = snap_span_to_segment(s_i, e_i, segs)
            starts.append(s_i); ends.append(e_i)

        # decode predictions
        for bi in range(j - off):
            real_len = lens[bi]
            toks = x[bi, :real_len].tolist()
            # reverse the 255-remap -> bytes
            byte_seq = bytes([255 if t == cfg.remap_255_to else int(t) for t in toks])
            s_i, e_i = starts[bi], ends[bi]
            pred = byte_seq[s_i:e_i+1].decode("utf-8", "ignore").strip()
            gold = golds[off + bi]
            q    = questions[off + bi]

            em  = _em(pred, gold)
            f1  = _f1(pred, gold)
            em_sum += em
            f1_sum += f1

            if shown < args.show:
                print(f"--- Example {shown+1} ---")
                print("Q:", q)
                print("Pred:", pred)
                print("Gold:", gold)
                shown += 1

            if args.dump_csv:
                rows_out.append({
                    "question": q,
                    "pred_raw": pred,
                    "pred_clean": _clean_value(pred),
                    "gold": gold,
                    "EM": f"{em:.0f}",
                    "F1": f"{f1:.4f}",
                })

    # --- report ---
    print("\nSamples evaluated:", n)
    print(f"Exact Match: {em_sum / max(1,n):.4f}")
    print(f"Token F1:     {f1_sum / max(1,n):.4f}")

    # optional CSV dump
    if args.dump_csv:
        os.makedirs(os.path.dirname(args.dump_csv), exist_ok=True) if os.path.dirname(args.dump_csv) else None
        with open(args.dump_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["question","pred_raw","pred_clean","gold","EM","F1"])
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"Wrote per-example results to: {args.dump_csv}")

if __name__ == "__main__":
    main()
