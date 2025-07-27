# utility/experiment_logger.py

import csv
import os
from datetime import datetime

LOG_FILE = "experiment_logs.csv"

def log_experiment(
    dataset_name,
    num_files,
    mask_prob,
    mask_strategy,
    embed_dim,
    num_layers,
    hidden_dim,
    num_heads,
    epochs,
    batch_size,
    learning_rate,
    accuracy,
    loss,
    notes=""
):
    log_path = os.path.join(os.getcwd(), LOG_FILE)
    fieldnames = [
        "timestamp", "dataset_name", "num_files", "mask_prob", "mask_strategy",
        "embed_dim", "num_layers", "hidden_dim", "num_heads",
        "epochs", "batch_size", "learning_rate",
        "accuracy", "loss", "notes"
    ]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name,
        "num_files": num_files,
        "mask_prob": mask_prob,
        "mask_strategy": mask_strategy,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "accuracy": round(accuracy * 100, 2),  # convert to percent
        "loss": round(loss, 4),
        "notes": notes
    }

    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"✅ Logged experiment to {LOG_FILE}")
