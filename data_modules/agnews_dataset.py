
import csv
from typing import List, Tuple, Optional
from torch.utils.data import Dataset

class AGNewsCSVDataset(Dataset):
    """
    Expects CSV files with header: text,label
    label should be 0..3 (four classes). If original 1..4, subtract 1 in preprocessing.
    """
    def __init__(self, path: str, max_len: int = 512, add_bos: bool = True, add_eos: bool = True):
        self.samples: List[Tuple[str, int]] = []
        self.max_len = max_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"]
                label = int(row["label"])
                self.samples.append((text, label))

    @staticmethod
    def text_to_bytes(text: str) -> List[int]:
        # Convert to UTF-8 bytes 0..255
        return list(text.encode("utf-8", errors="ignore"))

    def encode(self, text: str) -> List[int]:
        ids = self.text_to_bytes(text)
        if self.add_bos:
            ids = [257] + ids  # BOS
        if self.add_eos:
            ids = ids + [258]  # EOS
        # truncate
        ids = ids[: self.max_len]
        return ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        ids = self.encode(text)
        return {"input_ids": ids, "label": label}
