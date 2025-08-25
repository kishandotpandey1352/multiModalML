import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from configurations import config

MODALITY_TO_INDEX = {
    "text": 0,
    "audio": 1,
    "image": 2,
    "table": 3,
}

ALLOWED_EXTS = {
    "text": {".txt", ".text"},
    "audio": {".wav", ".mp3", ".flac", ".au", ".ogg"},
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".gif"},
    "table": {".txt", ".csv", ".tsv", ".xlsx", ".xls", ".parquet"}  # keep .xlsx but we stream safely
}

def _should_keep(path: Path, modality: str) -> bool:
    # Filter obvious non-files and hidden junk
    if not path.is_file():
        return False
    if path.name.startswith("._"):
        return False
    if modality in ALLOWED_EXTS:
        return path.suffix.lower() in ALLOWED_EXTS[modality]
    return True

class MultiModalDataset(Dataset):
    def __init__(self, data_path, modality=None, split="train", from_classifier=False, verbose=False):
        self.data_path = Path(data_path)
        self.files = []
        self.split = split
        self.from_classifier = from_classifier
        self.modality = modality
        self.seq_len = config.config["seq_len"]
        self.verbose = verbose

        # Validate modality
        if modality and modality not in MODALITY_TO_INDEX:
            raise ValueError(f"Unsupported modality: {modality}")

        # Gather files (apply extension filter & SAMPLE_SIZE limit)
        sample_limit = config.SAMPLE_SIZE  # may be -1 in your config
        def _limit(lst):
            if isinstance(sample_limit, int) and sample_limit > 0:
                return lst[:sample_limit]
            return lst  # keep all

        if modality:
            modality_path = self.data_path / modality
            all_files = [p for p in modality_path.rglob("*") if _should_keep(p, modality)]
            self.files = _limit(sorted(all_files))
        else:
            # Multimodal: gather per modality with per-modality limit
            for mod in MODALITY_TO_INDEX:
                mod_files = [p for p in (self.data_path / mod).rglob("*") if _should_keep(p, mod)]
                mod_files = _limit(sorted(mod_files))
                self.files.extend(mod_files)

        print(f"Found {len(self.files)} files for modality '{self.modality}' in {self.data_path}")

        # Extra: cap very large table files to avoid surprises (we still stream, but this avoids giant corpus proliferation)
        self.max_files_table = int(os.getenv("MAX_TABLE_FILES", "5000"))  # override via env if needed

    def __len__(self):
        return len(self.files)

    def _read_first_n_bytes(self, file_path: Path, n: int) -> bytes:
        """
        Stream only the first n bytes; never read the whole file.
        Works for all formats including .xlsx (we treat bytes uniformly).
        """
        with open(file_path, "rb") as f:
            return f.read(n)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        # Determine modality (explicit or inferred from folder name)
        if self.modality:
            modality_index = MODALITY_TO_INDEX[self.modality]
            modality_name = self.modality
        else:
            parent_folder = file_path.parent.name.lower()
            modality_index = MODALITY_TO_INDEX.get(parent_folder, 0)
            modality_name = parent_folder

        # Stream ONLY what we need (seq_len bytes) and avoid Python list conversion
        raw = self._read_first_n_bytes(file_path, self.seq_len)

        # Convert to a tensor efficiently (uint8 -> long)
        # Using bytearray avoids building a giant Python list
        byte_tensor = torch.tensor(bytearray(raw), dtype=torch.uint8).to(torch.long)

        # Pad or trim to seq_len (left as zeros)
        L = byte_tensor.numel()
        if L < self.seq_len:
            padded = torch.zeros(self.seq_len, dtype=torch.long)
            if L > 0:
                padded[:L] = byte_tensor
            byte_tensor = padded
        elif L > self.seq_len:
            byte_tensor = byte_tensor[:self.seq_len]

        # (Optional): downsample number of 'table' files per epoch without changing directory content
        if modality_name == "table" and idx >= self.max_files_table:
            # If you ever set MAX_TABLE_FILES to something small, this keeps __len__ truthful.
            # Alternatively, you could shuffle+slice self.files in __init__.
            pass

        # Quieter printing: only print occasionally to reduce I/O overhead
        if self.verbose or (idx % 200 == 0):
            print(f"File: {file_path}, modality: {modality_name}, index: {modality_index}")

        return {
            "byte_input": byte_tensor,                                  # (L,)
            "modality_index": torch.tensor(modality_index),             # ()
            "file_path": str(file_path)
        }
