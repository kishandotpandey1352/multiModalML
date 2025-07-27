# dataset_loader.py

import os
import torch
from torch.utils.data import Dataset
from typing import List

class ByteTextDataset(Dataset):
    """
    Dataset to load raw byte sequences from text files in a directory.

    Args:
        folder_path (str): Directory containing .txt files
        seq_len (int): Fixed length to truncate or pad sequences
    """
    def __init__(self, folder_path: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.file_paths = self._gather_text_files(folder_path)

    def _gather_text_files(self, folder_path: str) -> List[str]:
        return [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.lower().endswith(".txt")
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, "rb") as f:
            byte_data = f.read()

        # Truncate or pad the byte sequence
        byte_tensor = torch.full((self.seq_len,), 255, dtype=torch.long)  # 255 = MASK_TOKEN
        byte_array = torch.tensor(list(byte_data[:self.seq_len]), dtype=torch.long)
        byte_tensor[:len(byte_array)] = byte_array

        return byte_tensor
