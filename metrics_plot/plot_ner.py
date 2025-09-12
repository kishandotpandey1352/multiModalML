import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load your log
df = pd.read_csv("../logs/ner_log.csv")
df.columns = [c.strip().lower() for c in df.columns]
if "epoch" in df.columns:
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
for col in ["loss", "token_f1", "entity_f1"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
if "epoch" in df.columns and "split" in df.columns:
    df = df.sort_values(["epoch", "split"])

def plot_metric(df, metric, title, outfile):
    plt.figure(figsize=(8,5))
    if "split" in df.columns:
        for split, g in df.groupby("split"):
            g2 = g.dropna(subset=["epoch", metric])
            if not g2.empty:
                plt.plot(g2["epoch"].astype(int), g2[metric], marker="o", label=str(split))
    else:
        g2 = df.dropna(subset=["epoch", metric])
        if not g2.empty:
            plt.plot(g2["epoch"].astype(int), g2[metric], marker="o", label=metric)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    if "split" in df.columns:
        plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

# Make plots
plot_metric(df, "loss",      "NER Training Loss by Epoch",      "ner_loss.png")
plot_metric(df, "token_f1",  "Token-level F1 by Epoch",         "ner_token_f1.png")
plot_metric(df, "entity_f1", "Entity-level F1 by Epoch",        "ner_entity_f1.png")

# Best per split helper
def best_by_split(df, metric):
    out = []
    if not {"split","epoch",metric}.issubset(df.columns): 
        return out
    for split, g in df.groupby("split"):
        g2 = g.dropna(subset=[metric])
        if g2.empty:
            continue
        idx = g2[metric].idxmin() if metric == "loss" else g2[metric].idxmax()
        r = g2.loc[idx]
        out.append({"split": split, "epoch": int(r["epoch"]), metric: float(r[metric])})
    return out

print("Best loss:",      best_by_split(df, "loss"))
print("Best token_f1:",  best_by_split(df, "token_f1"))
print("Best entity_f1:", best_by_split(df, "entity_f1"))
