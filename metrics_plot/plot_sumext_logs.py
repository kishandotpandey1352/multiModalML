
#!/usr/bin/env python3
"""
plot_sumext_logs_v3.py  (no TEST curve)

- Understand logs with a 'split' column (train/val/test) where metrics share the same column.
- Segment multiple runs automatically (epoch resets to 1) and default to plotting the latest run only.
- Aggregate duplicates per epoch by taking the last value within a run and split.
- **This version plots only TRAIN and VAL** (ignores TEST rows).

Usage:
  python plot_sumext_logs_v3.py --csv sumext_log.csv --outdir plots --runs latest
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def add_run_id(df):
    """Assign a run_id that increments whenever epoch decreases (e.g., a new run)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    run_id = 0
    run_ids = [run_id]
    for i in range(1, len(df)):
        prev_e, cur_e = df.loc[i-1, "epoch"], df.loc[i, "epoch"]
        if cur_e < prev_e:  # new run if epoch resets or goes backwards
            run_id += 1
        run_ids.append(run_id)
    df["run_id"] = run_ids
    return df

def pivot_by_split(df, metric):
    """Return a dataframe with epoch as index and columns train/val (test dropped)."""
    df = df[df["split"].isin(["train", "val"])].copy()  # drop test rows
    if df.empty:
        return pd.DataFrame()
    df_p = (df[["epoch", "split", metric]]
              .dropna()
              .groupby(["split","epoch"], as_index=False)
              .last())
    piv = df_p.pivot(index="epoch", columns="split", values=metric).sort_index()
    # Ensure columns ordered train, val (if present)
    cols = [c for c in ["train","val"] if c in piv.columns]
    return piv[cols]

def plot_series(piv, title, ylabel, outfile):
    if piv.empty:
        return
    plt.figure()
    for col in piv.columns:  # only train/val due to pivot filter
        plt.plot(piv.index, piv[col], label=col)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

def main(args):
    df = pd.read_csv(args.csv)
    assert "epoch" in df.columns and "split" in df.columns, "CSV must have epoch and split columns."
    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))
    df = add_run_id(df)

    # choose run(s)
    if args.runs == "latest":
        last_run = df["run_id"].max()
        df = df[df["run_id"] == last_run].copy()
    elif args.runs == "all":
        pass
    else:
        rid = int(args.runs)
        df = df[df["run_id"] == rid].copy()

    os.makedirs(args.outdir, exist_ok=True)

    for metric, title, ylabel, fname in [
        ("loss", "Training/Validation Loss", "loss", "loss.png"),
        ("rougeL", "ROUGE-L F1", "F1", "rougeL_f1.png"),
        ("rouge1", "ROUGE-1 F1", "F1", "rouge1_f1.png"),
        ("rouge2", "ROUGE-2 F1", "F1", "rouge2_f1.png"),
    ]:
        if metric in df.columns:
            piv = pivot_by_split(df, metric)
            plot_series(piv, title, ylabel, os.path.join(args.outdir, fname))

    print("Plots saved to:", args.outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="sumext_log.csv")
    ap.add_argument("--outdir", type=str, default="sumext_plots_v3")
    ap.add_argument("--runs", type=str, default="latest", help="'latest' | 'all' | <run_id>")
    args = ap.parse_args()
    main(args)
