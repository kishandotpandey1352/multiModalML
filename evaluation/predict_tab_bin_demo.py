import torch
from data_modules.tab_hf import make_loader
from config_tab_bin import TabBinConfig
from tab_binary_head import TabBinaryHead
from train_tab_qa import load_shared_encoder, encode_to_sequence  # reuse helpers

@torch.no_grad()
def main():
    cfg = TabBinConfig()
    # paths you used during training
    cfg.init_checkpoint = "checkpoints/V1/model.pth"   # shared encoder
    head_ckpt = "checkpoints/tab_bin_adult/tab_bin_head.pth"  # your saved head

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_shared_encoder(cfg.init_checkpoint, device, cfg)
    encoder.eval()

    # infer hidden dim (D)
    fake_ids = torch.zeros(1, cfg.src_max_len, dtype=torch.long)
    fake_msk = torch.zeros(1, cfg.src_max_len, dtype=torch.long)
    D = encode_to_sequence(encoder, fake_ids, fake_msk, device).size(-1)

    head = TabBinaryHead(hidden_dim=D).to(device).eval()
    head.load_state_dict(torch.load(head_ckpt, map_location=device))

    # build a tiny eval loader from the same data module used in training
    loader = make_loader(cfg, cfg.split_val, batch_size=8)

    batch = next(iter(loader))
    x  = batch["input_ids"].to(device)
    m  = batch["attention_mask"].to(device)
    y  = batch["label"].float().to(device)

    seq = encode_to_sequence(encoder, x, m, device)    # [B,T,D]
    pooled = (seq * m.unsqueeze(-1)).sum(1) / (m.sum(1, keepdim=True).clamp_min(1))  # masked mean
    logit = head(pooled).squeeze(-1)                   # [B]
    prob  = torch.sigmoid(logit)

    print("probs:", prob.detach().cpu().numpy().round(3))
    print("golds:", y.detach().cpu().numpy().astype(int))
    print("preds:", (prob > 0.5).int().detach().cpu().numpy())

if __name__ == "__main__":
    main()
