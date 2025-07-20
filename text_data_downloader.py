# text_data_downloader.py
from datasets import load_dataset
import os

# === CONFIG ===
OUTPUT_DIR = "downloaded/dataset/text"
MAX_FILES = 200
DATASET_NAME = "cnn_dailymail"
CONFIG = "3.0.0"
TEXT_KEY = "article"

# === LOAD DATASET ===
print(f"Loading dataset: {DATASET_NAME} ({CONFIG})")
ds = load_dataset(DATASET_NAME, CONFIG, split="train")

# === WRITE FILES ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
count = 0
for example in ds:
    if TEXT_KEY not in example:
        continue
    text = example[TEXT_KEY].strip()
    if not text:
        continue

    out_path = os.path.join(OUTPUT_DIR, f"sample_{count}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    count += 1
    if count >= MAX_FILES:
        break

print(f"Saved {count} text files to {OUTPUT_DIR}")
