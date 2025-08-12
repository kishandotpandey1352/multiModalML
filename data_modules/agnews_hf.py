
import torch
from torch.utils.data import Dataset
import data_modules as hf_datasets  # <- explicit alias to avoid local package collision
import importlib, sys


try:
    m = sys.modules.get("datasets")
    if m is None or not hasattr(m, "load_dataset"):
        raise ImportError
    hf_datasets = m
except Exception:
    hf_datasets = importlib.import_module("datasets")

print("[debug] using datasets from:", getattr(hf_datasets, "__file__", "<unknown>"))
assert hasattr(hf_datasets, "load_dataset"), "HF datasets import failed"
class AGNewsHFBytes(Dataset):
    def __init__(self, split="train", max_len=1024, pad_token=255, encoding="utf-8"):
        self.ds = hf_datasets.load_dataset("ag_news", split=split)
        self.max_len = max_len
        self.pad_token = pad_token
        self.encoding = encoding

    def __len__(self):
        return len(self.ds)

    def _to_bytes(self, text: str):
        b = list(text.encode(self.encoding, errors="ignore"))
        b = b[: self.max_len]
        if len(b) < self.max_len:
            b = b + [self.pad_token] * (self.max_len - len(b))
        return torch.tensor(b, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.ds[idx]
        x = self._to_bytes(item["text"])
        y = int(item["label"])
        return x, torch.tensor(y, dtype=torch.long)
