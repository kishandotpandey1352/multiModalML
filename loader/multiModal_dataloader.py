
import os
import torch
from torch.utils.data import Dataset
from pathlib import Path

MODALITY_TO_INDEX = {
    "text": 0,
    "audio": 1,
    "image": 2,
    "table": 3,
}

class MultiModalDataset(Dataset):
    def __init__(self, data_path, modality=None, split="train", from_classifier=False):
        self.data_path = Path(data_path)
        self.files = list(self.data_path.glob("*"))
        self.split = split
        self.from_classifier = from_classifier
        self.modality = modality

        if modality:
            if modality not in MODALITY_TO_INDEX:
                raise ValueError(f"Unsupported modality: {modality}")
            self.modality_index = MODALITY_TO_INDEX[modality]
        elif from_classifier:
            raise ValueError("Classifier-based modality detection not implemented in this version.")
        else:
            raise ValueError("Modality must be specified unless from_classifier=True is implemented.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with open(file_path, "rb") as f:
            byte_data = f.read()

        byte_tensor = torch.tensor(list(byte_data), dtype=torch.long)

        if len(byte_tensor) < 512:
            padded = torch.full((512,), 0, dtype=torch.long)
            padded[:len(byte_tensor)] = byte_tensor
            byte_tensor = padded
        else:
            byte_tensor = byte_tensor[:512]

        return {
            "byte_input": byte_tensor,
            "modality_index": torch.tensor(self.modality_index, dtype=torch.long),
            "file_path": str(file_path)
        }
