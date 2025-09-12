import torch, pandas as pd
from config_tab_qa import TabQAConfig as TabIdxQAConfig  # reuse shape/pad knobs
from train_tab_qa import load_shared_encoder, encode_to_sequence
from heads.tabqa_span_head import TabQASpanHead
from data_modules.tabqa_synth import _serialize_row_for_tableqa, _bytes_to_ids

@torch.no_grad()
def main():
    cfg = TabIdxQAConfig()
    cfg.init_checkpoint = "checkpoints/V1/model.pth"
    head_ckpt = "checkpoints/tab_idxqa_adult/tab_idxqa_span_head.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = load_shared_encoder(cfg.init_checkpoint, device, cfg).eval()
    fake = torch.zeros(1, cfg.src_max_len, dtype=torch.long)
    D = encode_to_sequence(enc, fake, fake, device).size(-1)
    head = TabQASpanHead(hidden_dim=D).to(device).eval()
    head.load_state_dict(torch.load(head_ckpt, map_location=device))

    df = pd.read_csv("data/adult.csv")
    i = 17  # example 1-based index used during your training description
    col = "occupation"

    row = df.iloc[i-1].to_dict()   # convert to 0-based for pandas
    q = f"What is the value at row={i}, column={col}?"

    row_str, _ = _serialize_row_for_tableqa(row, add_schema=False, include_date_decompose=cfg.include_date_decompose)
    s = f"Q: {q}\n{row_str}"
    b = s.encode("utf-8", "ignore")
    ids, attn = _bytes_to_ids(b, cfg.src_max_len, cfg.pad_token, cfg.remap_255_to)

    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    m = torch.tensor(attn, dtype=torch.long).unsqueeze(0).to(device)

    seq = encode_to_sequence(enc, x, m, device)
    start_logits, end_logits = head(seq, m)
    start = int(start_logits.argmax(-1).item())
    end   = int(end_logits.argmax(-1).item())
    if end < start: end = start

    real_len = int(m[0].sum().item())
    payload = [cfg.remap_255_to if t == cfg.pad_token else int(t) for t in x[0][:real_len].tolist()]
    ans = bytes(payload)[start:end+1].decode("utf-8", "ignore").strip()

    print("Q:", q)
    print("Answer:", ans)
    print("(Reference)", col, "=", str(df.iloc[i-1][col]))

if __name__ == "__main__":
    main()
