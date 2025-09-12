# predict_tab_idxqa.py
import pandas as pd
import torch
from typing import Optional
from config_tab_idxqa import TabIdxQAConfig
from train_tab_qa import load_shared_encoder, encode_to_sequence
from heads.tabqa_span_head import TabQASpanHead
from data_modules.tab_idxqa import _bytes_to_ids, _serialize_row

@torch.no_grad()
def load_models(cfg: TabIdxQAConfig, head_ckpt_path: str, device: Optional[str]=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = load_shared_encoder(cfg.init_checkpoint, device, cfg)
    # infer hidden dim
    fake_ids = torch.zeros(1, cfg.src_max_len, dtype=torch.long, device=device)
    fake_mask = torch.zeros(1, cfg.src_max_len, dtype=torch.long, device=device)
    D = encode_to_sequence(enc, fake_ids, fake_mask, device).size(-1)
    head = TabQASpanHead(hidden_dim=D).to(device)
    head.load_state_dict(torch.load(head_ckpt_path, map_location=device))
    enc.eval(); head.eval()
    return enc, head, device

@torch.no_grad()
def predict_index(cfg: TabIdxQAConfig, df: pd.DataFrame, row_idx: int, col, enc, head, device: str):
    # build question
    row = df.iloc[row_idx].to_dict()
    if isinstance(col, int):
        col_name = df.columns[col]
        q = f"Q: What is the value at row={row_idx + (cfg.index_base or 0)}, col={col + (cfg.index_base or 0)}?"
    else:
        col_name = str(col)
        q = f"Q: What is the value at row={row_idx + (cfg.index_base or 0)}, column={col_name}?"

    row_ser = _serialize_row(row, cfg.add_schema_header, cfg.include_date_decompose)
    serialized = f"{q}\n{row_ser}"
    b = serialized.encode("utf-8", "ignore")
    ids, attn = _bytes_to_ids(b, cfg.src_max_len, cfg.pad_token, cfg.remap_255_to)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    m = torch.tensor(attn, dtype=torch.long, device=device).unsqueeze(0)

    seq = encode_to_sequence(enc, x, m, device)
    start_logits, end_logits = head(seq, m)
    s = int(start_logits.argmax(dim=-1).item())
    e = int(end_logits.argmax(dim=-1).item())
    if e < s: e = s
    # reconstruct bytes (unmap PAD)
    valid = int(m[0].sum().item())
    raw_bytes = bytes([(cfg.remap_255_to if t == cfg.pad_token else t) for t in x[0, :valid].tolist()])
    ans = raw_bytes[s:e+1].decode("utf-8", "ignore").strip()
    return {"answer_text": ans, "start": s, "end": e, "question": q}

if __name__ == "__main__":
    cfg = TabIdxQAConfig()
    enc, head, device = load_models(cfg, "checkpoints/tab_idxqa_adult/tab_idxqa_span_head.pth")
    df = pd.read_csv(cfg.csv_path, delimiter=cfg.delimiter)
    print(predict_index(cfg, df, row_idx=0, col="occupation", enc=enc, head=head, device=device))
