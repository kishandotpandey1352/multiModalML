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
        self.files = [f for f in self.data_path.glob("**/*") if f.is_file()]
        self.split = split
        self.from_classifier = from_classifier
        self.modality = modality

        # Validate modalities
        if modality and modality not in MODALITY_TO_INDEX:
            raise ValueError(f"Unsupported modality: {modality}")

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

        # Determine modality
        if self.modality:
            modality_name = self.modality
            modality_index = MODALITY_TO_INDEX[self.modality]
        else:
            modality_name = file_path.parent.name.lower()
            modality_index = MODALITY_TO_INDEX.get(modality_name, 0)  # default to 0 if unknown

        # Print debug info
        print(f"File: {file_path}, modality: {modality_name}, index: {modality_index}")

        return {
            "byte_input": byte_tensor,
            "modality_index": torch.tensor(modality_index, dtype=torch.long),
            "file_path": str(file_path)
        }
