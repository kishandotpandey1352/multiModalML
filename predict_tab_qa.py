# predict_tab_qa.py
import re
import pandas as pd
import torch
from utils.config_tab_qa import TabQAConfig
from train_tab_qa import load_shared_encoder, encode_to_sequence
from heads.tabqa_span_head import TabQASpanHead
from data_modules.tabqa_synth import _serialize_row_for_tableqa, _bytes_to_ids

# ---------- helpers ----------
def _parse_selectors_from_query(query: str, columns) -> dict:
    """
    Extracts simple 'col = value' pairs used in your synthetic queries.
    Handles hyphenated column names (e.g., 'marital-status').
    """
    cols = set(map(str, columns))
    pairs = {}
    # find all tokens like: <col> = <value> (stop at 'and' or end/punct)
    for m in re.finditer(r'([A-Za-z0-9_-]+)\s*=\s*([^&\n]+?)(?:\s+and\b|[?]|$)', query):
        k = m.group(1).strip()
        v = m.group(2).strip().strip('"\'')
        if k in cols:
            pairs[k] = v
    return pairs

def _pick_row_by_query(df: pd.DataFrame, query: str) -> dict:
    sels = _parse_selectors_from_query(query, df.columns)
    if not sels:
        # fallback: first row
        return df.iloc[0].to_dict()
    mask = pd.Series(True, index=df.index)
    for k, v in sels.items():
        if k in df.columns:
            mask &= (df[k].astype(str).str.strip() == str(v).strip())
    if mask.any():
        return df.loc[mask].iloc[0].to_dict()
    # fallback if no match
    return df.iloc[0].to_dict()

@torch.no_grad()
def load_models(cfg: TabQAConfig, head_ckpt_path: str, device: str=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = load_shared_encoder(cfg.init_checkpoint, device, cfg)
    # infer hidden size
    fake_ids = torch.zeros(1, cfg.src_max_len, dtype=torch.long)
    fake_mask = torch.zeros(1, cfg.src_max_len, dtype=torch.long)
    D = encode_to_sequence(enc, fake_ids, fake_mask, device).size(-1)
    head = TabQASpanHead(hidden_dim=D).to(device)
    head.load_state_dict(torch.load(head_ckpt_path, map_location=device))
    enc.eval(); head.eval()
    return enc, head, device

@torch.no_grad()
def predict_query(cfg: TabQAConfig, df: pd.DataFrame, query: str, encoder, head, device: str):
    # 1) pick a row that matches query conditions
    row = _pick_row_by_query(df, query)
    # 2) serialize exactly like training (query + typed row)
    serial_row, _ = _serialize_row_for_tableqa(row, cfg.add_schema_header, cfg.include_date_decompose)
    s = f"Q: {query}\n{serial_row}"
    b = s.encode("utf-8", "ignore")
    # 3) bytes -> ids/mask
    ids, attn = _bytes_to_ids(b, cfg.src_max_len, cfg.pad_token, cfg.remap_255_to)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    m = torch.tensor(attn, dtype=torch.long).unsqueeze(0).to(device)
    # 4) encoder + span head
    seq = encode_to_sequence(encoder, x, m, device)
    start_logits, end_logits = head(seq, m)
    start = int(start_logits.argmax(dim=-1).item())
    end   = int(end_logits.argmax(dim=-1).item())
    if end < start: end = start
    # 5) decode bytes back to text
    real_len = int(m[0].sum().item())
    # reverse PAD remap for decoding
    byte_seq = [(cfg.remap_255_to if t == cfg.pad_token else int(t)) for t in x[0][:real_len].tolist()]
    bb = bytes(byte_seq)
    ans = bb[start:end+1].decode("utf-8", "ignore").strip()
    return {"answer_text": ans, "start": start, "end": end, "matched_selectors": _parse_selectors_from_query(query, df.columns)}

if __name__ == "__main__":
    # --- config ---
    cfg = TabQAConfig()
    cfg.init_checkpoint = "checkpoints/V1/model.pth"
    cfg.src_max_len = 1536  # keep consistent with training
    head_ckpt = "checkpoints/tab_qa_adult/tab_qa_span_head.pth"   # match your save_dir/save_head_as

    # --- load ---
    enc, head, device = load_models(cfg, head_ckpt)

    # --- data ---
    df = pd.read_csv("data/adult.csv")

    # --- query (example consistent with your training templates) ---
    query = "What is the hours-per-week for the row where age = 39 and education = Bachelors?"

    out = predict_query(cfg, df, query, enc, head, device)
    print(out)
