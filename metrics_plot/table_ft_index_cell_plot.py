import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Locate the CSV ---
# candidates = [
#     Path("logs\tab_idxqa_adult.csv"),       # your project path
#     Path("data\adult.csv"),  # fallback (for this workspace)
# ]
# csv_path = next((p for p in candidates if p.exists()), None)
# if csv_path is None:
#     raise FileNotFoundError("logs\\tab_idxqa_adult.csv not found")

# --- Load ---
df = pd.read_csv("../logs/tab_idxqa_adult.csv")
df.columns = [c.strip() for c in df.columns]
lower = {c.lower(): c for c in df.columns}
def col(*names):
    for n in names:
        if n in lower: return lower[n]
    raise KeyError(f"Missing columns; have {list(df.columns)}")

epoch_c = col("epoch")
train_c = col("train_loss","train","trainloss")
val_c   = col("val_loss","valid_loss","validation")
em_c    = col("em","exact_match","exactmatch")
f1_c    = col("f1","char_f1","char-f1")
lr_c    = col("lr","learning_rate","learningrate")

df = df.sort_values(epoch_c).reset_index(drop=True)

# --- Best checkpoints (for your report) ---
best_val_idx = df[val_c].idxmin()
best_em_idx  = df[em_c].idxmax()
best_f1_idx  = df[f1_c].idxmax()

print("Best by val_loss:", dict(epoch=int(df.loc[best_val_idx, epoch_c]),
                                val_loss=float(df.loc[best_val_idx, val_c]),
                                train_loss=float(df.loc[best_val_idx, train_c]),
                                EM=float(df.loc[best_val_idx, em_c]),
                                F1=float(df.loc[best_val_idx, f1_c])))
print("Best by EM:",      dict(epoch=int(df.loc[best_em_idx, epoch_c]),
                                EM=float(df.loc[best_em_idx, em_c]),
                                F1=float(df.loc[best_em_idx, f1_c]),
                                val_loss=float(df.loc[best_em_idx, val_c])))
print("Best by F1:",      dict(epoch=int(df.loc[best_f1_idx, epoch_c]),
                                F1=float(df.loc[best_f1_idx, f1_c]),
                                EM=float(df.loc[best_f1_idx, em_c]),
                                val_loss=float(df.loc[best_f1_idx, val_c])))

# --- Plots (1 metric per figure; no custom colors) ---
out_dir = Path("plots"); out_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(6,4), dpi=160)
plt.plot(df[epoch_c], df[train_c], label="train_loss")
plt.plot(df[epoch_c], df[val_c],   label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Idx→Cell QA: Train vs Val Loss")
plt.legend(); plt.tight_layout()
plt.savefig(out_dir/"idxqa_loss.png"); plt.close()

plt.figure(figsize=(6,4), dpi=160)
plt.plot(df[epoch_c], df[em_c], marker="o")
plt.xlabel("Epoch"); plt.ylabel("Exact Match (EM)"); plt.title("Idx→Cell QA: EM by Epoch")
plt.tight_layout()
plt.savefig(out_dir/"idxqa_em.png"); plt.close()

plt.figure(figsize=(6,4), dpi=160)
plt.plot(df[epoch_c], df[f1_c], marker="o")
plt.xlabel("Epoch"); plt.ylabel("Char-F1"); plt.title("Idx→Cell QA: Char-F1 by Epoch")
plt.tight_layout()
plt.savefig(out_dir/"idxqa_f1.png"); plt.close()

plt.figure(figsize=(6,4), dpi=160)
plt.plot(df[epoch_c], df[lr_c], marker="o")
plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.title("Idx→Cell QA: Learning Rate by Epoch")
plt.tight_layout()
plt.savefig(out_dir/"idxqa_lr.png"); plt.close()

print("Saved plots in:", out_dir.resolve())
