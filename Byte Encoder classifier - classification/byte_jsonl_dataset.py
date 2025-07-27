import torch
from torch.utils.data import Dataset
import json
import random

LABEL_MAP = {'audio': 0, 'image': 1, 'text': 2, 'table': 3}

class JsonlByteDataset(Dataset):
    def __init__(self, jsonl_path, input_len=2048, random_chunking=True):
        self.samples = []
        self.input_len = input_len
        self.random_chunking = random_chunking

        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                self.samples.append((entry['file_path'], LABEL_MAP[entry['label']]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        try:
            with open(fpath, 'rb') as f:
                f.seek(0, 2)  # move to end to get file size
                file_size = f.tell()

                if file_size <= self.input_len:
                    f.seek(0)
                    raw_bytes = f.read()
                else:
                    if self.random_chunking:
                        offset = random.randint(0, file_size - self.input_len)
                        f.seek(offset)
                    else:
                        f.seek(0)
                    raw_bytes = f.read(self.input_len)
        except Exception:
            raw_bytes = bytes([0] * self.input_len)

        x = torch.tensor(list(raw_bytes), dtype=torch.uint8).float() / 255.0
        if x.size(0) < self.input_len:
            pad = torch.zeros(self.input_len - x.size(0))
            x = torch.cat([x, pad])
        return x, label
