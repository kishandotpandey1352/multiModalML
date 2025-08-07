import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class AGNewsByteDataset(Dataset):
    def __init__(self, split="train", max_len=256):
        """
        Args:
            split: 'train' or 'test'
            max_len: max number of bytes (characters) to keep per example
        """
        self.dataset = load_dataset("ag_news", split=split)
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        label = item["label"]  # Label: 0 to 3 (World, Sports, Business, Sci/Tech)

        # Encode to bytes (truncate or pad)
        byte_input = torch.zeros(self.max_len, dtype=torch.long)
        encoded = [ord(c) for c in text[:self.max_len]]
        byte_input[:len(encoded)] = torch.tensor(encoded, dtype=torch.long)

        return {
                    "byte_input": byte_input.clone().detach().to(dtype=torch.long),
                    "label": torch.tensor(label, dtype=torch.long)
                }
