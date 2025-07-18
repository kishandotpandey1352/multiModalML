# byte_dataloader.py
import os
import torch
from torch.utils.data import Dataset

class ByteDataset(Dataset):
    def __init__(self, root_dir, input_len=2048):
        self.root_dir = root_dir
        self.input_len = input_len
        self.samples = []
        self.label_map = {'audio': 0, 'image': 1, 'text': 2, 'table': 3}

        for label_name, label in self.label_map.items():
            folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        try:
            with open(fpath, 'rb') as f:
                raw_bytes = f.read(self.input_len)
        except Exception as e:
            raw_bytes = bytes([0] * self.input_len)

        byte_tensor = torch.tensor(list(raw_bytes), dtype=torch.uint8).float() / 255.0
        if byte_tensor.size(0) < self.input_len:
            pad = torch.zeros(self.input_len - byte_tensor.size(0))
            byte_tensor = torch.cat([byte_tensor, pad])
        return byte_tensor, label