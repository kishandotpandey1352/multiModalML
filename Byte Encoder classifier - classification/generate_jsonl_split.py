import os
import json
import random
from pathlib import Path

# Configuration
DATASET_ROOT = "dataset"
OUTPUT_DIR = "./"  
SPLIT_RATIOS = [0.7, 0.15, 0.15]  # train, val, test
MODALITY_LABELS = ["audio", "image", "text", "table"]

# Collect all file paths and their labels
samples = []
for label in MODALITY_LABELS:
    class_dir = Path(DATASET_ROOT) / label
    if not class_dir.exists():
        continue
    for file in class_dir.iterdir():
        if file.is_file():
            samples.append({"file_path": str(file), "label": label})

# Shuffle for random split
random.seed(42)
random.shuffle(samples)

# Split samples
total = len(samples)
train_end = int(SPLIT_RATIOS[0] * total)
val_end = train_end + int(SPLIT_RATIOS[1] * total)

splits = {
    "modality_train.jsonl": samples[:train_end],
    "modality_val.jsonl": samples[train_end:val_end],
    "modality_test.jsonl": samples[val_end:]
}

# Write .jsonl files
for fname, entries in splits.items():
    out_path = Path(OUTPUT_DIR) / fname
    with open(out_path, "w") as f:
        for entry in entries:
            json.dump(entry, f)
            f.write("\n")
    print(f"Wrote {len(entries)} samples to {fname}")