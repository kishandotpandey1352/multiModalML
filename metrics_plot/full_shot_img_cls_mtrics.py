import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

log_path = Path("../logs/img_cls_full.csv")

# Let pandas use the actual header row from your CSV
full_shot = pd.read_csv(log_path)

# Quick sanity print
print("Columns:", list(full_shot.columns))

# Create a 1x2 figure: accuracy (left), loss (right)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Accuracy subplot ---
axes[0].plot(full_shot["epoch"], full_shot["train_acc"], label="Train Acc")
axes[0].plot(full_shot["epoch"], full_shot["val_acc"],   label="Val Acc")

# Plot Top‑5 if present
if "val_acc5" in full_shot.columns:
    axes[0].plot(full_shot["epoch"], full_shot["val_acc5"], label="Val Top‑5 Acc")

axes[0].set_title("Full data Training — Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(alpha=0.3)

# --- Loss subplot ---
axes[1].plot(full_shot["epoch"], full_shot["train_loss"], label="Train Loss")
axes[1].plot(full_shot["epoch"], full_shot["val_loss"],   label="Val Loss")
axes[1].set_title("Full data Training — Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("full_acc_loss.png", dpi=200)
plt.close(fig)
