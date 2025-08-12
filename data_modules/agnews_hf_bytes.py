
import torch
from torch.utils.data import Dataset
from data_modules import load_dataset

class AGNewsBytesHF(Dataset):
    def __init__(self, split="train", max_len=1024, pad_token=255, encoding="utf-8"):
        self.ds = load_dataset("ag_news", split=split)
        self.max_len = max_len
        self.pad_token = pad_token
        self.encoding = encoding

    def __len__(self):
        return len(self.ds)

    def _to_bytes(self, text: str):
        # Use the exact byte representation (utf-8) like pre-training on raw files
        b = list(text.encode(self.encoding, errors="ignore"))
        b = b[: self.max_len]
        # left-pad with pad_token? We'll right-pad to fixed length with pad_token (=255)
        if len(b) < self.max_len:
            b = b + [self.pad_token] * (self.max_len - len(b))
        return torch.tensor(b, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.ds[idx]
        x = self._to_bytes(item["text"])
        y = int(item["label"])
        return x, torch.tensor(y, dtype=torch.long)
