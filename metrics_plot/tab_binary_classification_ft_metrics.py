import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

log_path = Path("../logs/tab_bin_adult_sklearn.csv")
df = pd.read_csv(log_path)
# Prepare output dir
out_dir = Path("./")
fig1_path = out_dir / "tab_bin_loss_curves.png"
fig2_path = out_dir / "tab_bin_val_metrics.png"

# --------- Figure 1: Loss Curves ---------
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss (Tabular Binary Head)')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig(fig1_path)
plt.show()

# --------- Figure 2: Validation Metrics ---------
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['val_auc'], label='Val AUC', marker='o')
plt.plot(df['epoch'], df['val_acc'], label='Val Accuracy', marker='s')
plt.plot(df['epoch'], df['val_f1'], label='Val F1', marker='^')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation Metrics over Epochs')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig(fig2_path)
plt.show()

# Compute summary stats
summary = {}
summary['num_epochs'] = int(df['epoch'].max())
summary['best_auc'] = float(df['val_auc'].max())
summary['best_auc_epoch'] = int(df.loc[df['val_auc'].idxmax(), 'epoch'])
summary['best_f1'] = float(df['val_f1'].max())
summary['best_f1_epoch'] = int(df.loc[df['val_f1'].idxmax(), 'epoch'])
summary['best_acc'] = float(df['val_acc'].max())
summary['best_acc_epoch'] = int(df.loc[df['val_acc'].idxmax(), 'epoch'])
summary['min_val_loss'] = float(df['val_loss'].min())
summary['min_val_loss_epoch'] = int(df.loc[df['val_loss'].idxmin(), 'epoch'])
summary['final_epoch'] = int(df['epoch'].iloc[-1])
summary['final_train_loss'] = float(df['train_loss'].iloc[-1])
summary['final_val_loss'] = float(df['val_loss'].iloc[-1])
summary

(fig1_path.as_posix(), fig2_path.as_posix(), summary)
