# top of file
import os, csv, torch, random
from datetime import datetime
from torch import nn, optim
from utility.span_masking import span_mask_input
from loader.multiModal_dataloader import MultiModalDataset
from encoder.byte_encoder import ByteEncoder
from decoders.span_boundary_decoder import SpanBoundaryDecoder
from trainer.continual import ReplayBuffer, feature_distillation_loss, compute_fisher_diag, ewc_penalty
from configurations.config import config
from torch.utils.data import DataLoader, random_split
import copy

device = torch.device(config.get("device","cuda") if torch.cuda.is_available() else "cpu")

# ==== Model ====
ckpt_path = config["checkpoint_path"]
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
encoder = ByteEncoder(config).to(device)
decoder = SpanBoundaryDecoder(config).to(device)
old_encoder, old_decoder = None, None
prev_params, fisher = None, None   # for optional EWC

if os.path.exists(ckpt_path):
    print(f"\n✅ Loading from {ckpt_path}")
    ckpt_data = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt_data['encoder'])
    decoder.load_state_dict(ckpt_data['decoder'])
    # keep a frozen copy for distillation
    old_encoder = copy.deepcopy(encoder).eval().requires_grad_(False)
    old_decoder = copy.deepcopy(decoder).eval().requires_grad_(False)
    # (optional) Fisher + prev params for EWC
    if "fisher" in ckpt_data and "prev_params" in ckpt_data:
        fisher = {k: v.to(device) for k, v in ckpt_data["fisher"].items()}
        prev_params = {k: p.to(device) for k, p in ckpt_data["prev_params"].items()}
else:
    print("\n🚨 No checkpoint found — starting from scratch")

optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=config["lr"])
criterion = nn.CrossEntropyLoss(ignore_index=config["ignore_index"])

# ==== Data ====
full_dataset = MultiModalDataset(
    data_path=config["data_path"], modality=config["modality"], split='train'
)
train_size = int(0.9 * len(full_dataset)) if len(full_dataset) > 1 else len(full_dataset)
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# ==== Replay buffer ====
replay = ReplayBuffer(max_per_modality=256)

# ==== Training ====
best_loss = float('inf')
no_improve = 0
patience = config["early_stopping_patience"]
epochs = config["epochs"]
modality_name = config["modality"]

for epoch in range(1, epochs + 1):
    encoder.train(); decoder.train()
    total_train_loss = 0; train_batches = 0

    for batch in train_loader:
        byte_input = batch["byte_input"].to(device)
        mod_idx = batch["modality_index"].to(device)

        # add to replay
        if "file_path" in batch:
            replay.add_batch(modality_name, batch["file_path"], batch["byte_input"])

        # mask current
        masked, labels = span_mask_input(byte_input, mask_prob=config["mask_prob"], max_span_length=config["mask_span_length"])
        masked, labels = masked.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # forward current
        hidden = encoder(masked, mod_idx)  # (B,L,D)

        # collect boundary supervision
        loss_sup = 0.0; count = 0
        for b in range(hidden.size(0)):
            lbl = labels[b]
            inp = byte_input[b]
            L = lbl.size(0)
            i = 0
            left_batch=[]; right_batch=[]; rel_pos=[]; targets=[]
            while i < L:
                if lbl[i] != config["ignore_index"]:
                    start=i
                    while i < L and lbl[i] != config["ignore_index"]:
                        i+=1
                    end=i-1
                    left  = hidden[b, start-1] if start>0 else torch.zeros_like(hidden[b,0])
                    right = hidden[b, end+1]   if end+1<hidden.size(1) else torch.zeros_like(hidden[b,0])
                    left_batch.append(left); right_batch.append(right)
                    rel_pos.append(end-start+1)
                    targets.append(inp[start])
                i+=1
            if targets:
                left_batch = torch.stack(left_batch)
                right_batch= torch.stack(right_batch)
                rel_pos   = torch.tensor(rel_pos, device=device)
                targets   = torch.tensor(targets, device=device)
                logits = decoder(left_batch, right_batch, rel_pos)
                loss_sup = loss_sup + criterion(logits, targets)
                count += 1
        if count>0:
            loss_sup = loss_sup / count

        # ===== Replay + distillation =====
        loss_distill = torch.tensor(0.0, device=device)
        if old_encoder is not None:
            sample = replay.sample(batch_size=1)
            if sample is not None:
                _, _, bt = sample[0]
                bt = bt.unsqueeze(0).to(device)
                # Mask in the same way for fairness
                masked_r, _ = span_mask_input(bt, mask_prob=config["mask_prob"], max_span_length=config["mask_span_length"])
                masked_r = masked_r.to(device)
                with torch.no_grad():
                    old_feat = old_encoder(masked_r, mod_idx.new_zeros(1))  # any valid mod idx for old feat
                new_feat = encoder(masked_r, mod_idx.new_zeros(1))
                loss_distill = feature_distillation_loss(new_feat, old_feat)

        # ===== Optional EWC =====
        loss_ewc = torch.tensor(0.0, device=device)
        if prev_params is not None and fisher is not None:
            loss_ewc = ewc_penalty(encoder, prev_params, fisher, lam=50.0)

        loss = loss_sup + 0.5*loss_distill + loss_ewc
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item(); train_batches += 1

    avg_train_loss = total_train_loss / max(train_batches,1)

    # ==== Validation (supervised only) ====
    encoder.eval(); decoder.eval()
    total_val_loss=0; val_batches=0
    with torch.no_grad():
        for batch in val_loader:
            byte_input = batch['byte_input'].to(device)
            mod_idx = batch['modality_index'].to(device)
            masked, labels = span_mask_input(byte_input, config["mask_prob"], config["mask_span_length"])
            masked, labels = masked.to(device), labels.to(device)
            hidden = encoder(masked, mod_idx)
            loss_sup=0; count=0
            for b in range(hidden.size(0)):
                lbl=labels[b]; inp=byte_input[b]
                L=lbl.size(0); i=0
                lb=[]; rb=[]; rp=[]; tgt=[]
                while i<L:
                    if lbl[i]!=config["ignore_index"]:
                        s=i
                        while i<L and lbl[i]!=config["ignore_index"]:
                            i+=1
                        e=i-1
                        lb.append(hidden[b, s-1] if s>0 else torch.zeros_like(hidden[b,0]))
                        rb.append(hidden[b, e+1] if e+1<hidden.size(1) else torch.zeros_like(hidden[b,0]))
                        rp.append(e-s+1)
                        tgt.append(inp[s])
                    i+=1
                if tgt:
                    lb=torch.stack(lb); rb=torch.stack(rb)
                    rp=torch.tensor(rp, device=device); tgt=torch.tensor(tgt, device=device)
                    logits=decoder(lb, rb, rp)
                    loss_sup += criterion(logits, tgt); count+=1
            if count>0:
                total_val_loss += (loss_sup/count).item(); val_batches+=1
    avg_val_loss = total_val_loss / max(val_batches,1)

    print(f"Epoch {epoch}: train {avg_train_loss:.4f} | val {avg_val_loss:.4f}")

    # Save best + (optional) Fisher for EWC at the end of this modality
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss; no_improve = 0
        save_dict = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict()
        }
        # compute Fisher on this modality to protect it next time
        fisher = compute_fisher_diag(encoder, (x for x in train_dataset), device, n_steps=200)
        prev_params = {n: p.detach().clone().cpu() for n,p in encoder.named_parameters() if p.requires_grad}
        save_dict["fisher"] = {k: v.cpu() for k,v in fisher.items()}
        save_dict["prev_params"] = prev_params
        torch.save(save_dict, ckpt_path)
        print("✅ Checkpoint + Fisher saved.")
    else:
        no_improve += 1
        print(f"⚠️ No improvement for {no_improve} epoch(s)")
        if no_improve >= patience:
            print("⏹️ Early stopping.")
            break
