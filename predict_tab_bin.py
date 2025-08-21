# predict_tab_bin.py
import torch
import torch.nn.functional as F

from utils.config_tab_bin import TabBinConfig
from train_tab_bin import load_shared_encoder, encode_to_sequence
from heads.tab_binary_head import TabBinaryHead
from data_modules.tab_hf import _serialize_row, _bytes_to_ids  # reuse your adapter

@torch.no_grad()
def load_models(cfg: TabBinConfig, head_ckpt_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # 1) Load encoder (frozen)
    encoder = load_shared_encoder(cfg.init_checkpoint, device, cfg)
    encoder.eval()

    # 2) Infer hidden dim from a tiny fake input
    fake_ids = torch.zeros(1, cfg.src_max_len, dtype=torch.long)
    fake_mask = torch.zeros(1, cfg.src_max_len, dtype=torch.long)
    seq = encode_to_sequence(encoder, fake_ids, fake_mask, device)
    hidden_dim = seq.size(-1)

    # 3) Load head
    head = TabBinaryHead(hidden_dim=hidden_dim).to(device).eval()
    state = torch.load(head_ckpt_path, map_location=device)
    head.load_state_dict(state)
    return encoder, head, device

def row_to_batch_tensors(row: dict, cfg: TabBinConfig):
    s = _serialize_row(row, cfg.add_schema_header, cfg.include_date_decompose)
    b = s.encode("utf-8", errors="ignore")
    ids, attn = _bytes_to_ids(b, cfg.src_max_len, cfg.pad_token, cfg.remap_255_to)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)     # [1, L]
    m = torch.tensor(attn, dtype=torch.long).unsqueeze(0)    # [1, L]
    return x, m

@torch.no_grad()
def predict_one(row: dict, cfg: TabBinConfig, encoder, head, device: str, threshold: float = 0.5):
    x, m = row_to_batch_tensors(row, cfg)
    x, m = x.to(device), m.to(device)

    seq = encode_to_sequence(encoder, x, m, device)     # [1, T, D]
    logit = head(seq, m).squeeze(0)                     # []
    prob = torch.sigmoid(logit).item()

    # Map back to label strings
    pos = cfg.positive_label if cfg.positive_label is not None else "1"
    neg = "not " + str(pos) if cfg.positive_label is not None else "0"
    pred_label = pos if prob >= threshold else neg

    return {"prob_pos": prob, "pred_label": pred_label}

if __name__ == "__main__":
    # ----- Configure -----
    cfg = TabBinConfig()
    # Match your training settings
    cfg.dataset_name   = "scikit-learn/adult-census-income"
    cfg.label_column   = "income"
    cfg.positive_label = ">50K"
    cfg.src_max_len    = 1024
    cfg.pad_token      = 255
    cfg.remap_255_to   = 254

    head_ckpt = "checkpoints/tab_bin_adult_sklearn/tab_bin_head.pth"  # adjust if different

    # ----- Load models -----
    encoder, head, device = load_models(cfg, head_ckpt)

    # ----- Example row (fill with real values/columns from your CSV) -----
    sample = {
        "age": 39,
        "workclass": "Private",
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        # include any other columns present in the dataset row
    }

    out = predict_one(sample, cfg, encoder, head, device, threshold=0.5)
    print(out)  # {'prob_pos': 0.xxx, 'pred_label': '>50K' or '<=50K'}
