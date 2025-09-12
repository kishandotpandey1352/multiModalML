# data_modules/tab_idxqa.py
import random, re
from typing import Optional, Dict, Any
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader

def _bytes_to_ids(b: bytes, max_len: int, pad_token: int, remap_255_to: int):
    ids = list(b[:max_len])
    attn = [1] * len(ids)
    if len(ids) < max_len:
        padn = max_len - len(ids)
        ids += [pad_token] * padn
        attn += [0] * padn
    # reserve 255 for PAD
    ids = [remap_255_to if t == 255 else t for t in ids]
    return ids, attn

def _serialize_row(row: Dict[str, Any], add_schema_header: bool, include_date_decompose: bool):
    # Minimal typed row serialization compatible with your previous QA setup
    parts = []
    for k, v in row.items():
        typ = "NUM" if isinstance(v, (int, float)) else "STR"
        parts.append(f"{k}=<{typ}:{v}>")
    row_str = " | ".join(parts)
    s = f"ROW: {row_str}"
    return s

def _find_value_span_in_row(serialized: str, col_name: str, value: Any):
    # locate "<...:VALUE>" within "col_name=<TYPE:VALUE>"
    value_str = str(value)
    pat = re.compile(re.escape(f"{col_name}=") + r"<[^:>]*:(.*?)>")
    m = None
    for match in pat.finditer(serialized):
        if match.group(1) == value_str:
            m = match
            break
        if m is None:
            m = match
    if m is None:
        # Fallback raw search
        start_char = serialized.find(value_str)
        if start_char < 0:
            raise RuntimeError(f"Value span not found for {col_name}={value_str}")
        end_char = start_char + len(value_str) - 1
    else:
        start_char = m.start(1)
        end_char = m.end(1) - 1

    # map char offsets → byte offsets
    start_byte = len(serialized[:start_char].encode("utf-8", "ignore"))
    end_byte = start_byte + len(value_str.encode("utf-8", "ignore")) - 1
    return start_byte, end_byte

class TableIndexQAStream(IterableDataset):
    """
    Samples (row_idx, col) queries and returns:
    input_ids [T], attention_mask [T], start_idx, end_idx, win_start, win_end, answer_text
    """
    def __init__(self, cfg, split: str, dataframe: Optional[pd.DataFrame]=None):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.df = dataframe if dataframe is not None else pd.read_csv(cfg.csv_path, delimiter=cfg.delimiter)
        if cfg.max_rows_per_table is not None:
            self.df = self.df.head(cfg.max_rows_per_table)
        self.cols = list(self.df.columns)

    def _build_index_example(self):
        cfg = self.cfg
        df = self.df
        n_rows = len(df); n_cols = len(self.cols)
        row_idx = random.randrange(n_rows)
        col_idx = random.randrange(n_cols)
        col_name = self.cols[col_idx]
        row = df.iloc[row_idx].to_dict()

        # question: by name or by numeric index
        hrow = row_idx + (cfg.index_base or 0)
        if cfg.use_col_index:
            hcol = col_idx + (cfg.index_base or 0)
            q = f"Q: What is the value at row={hrow}, col={hcol}?"
        else:
            q = f"Q: What is the value at row={hrow}, column={col_name}?"

        row_ser = _serialize_row(row, cfg.add_schema_header, cfg.include_date_decompose)
        serialized = f"{q}\n{row_ser}"
        b = serialized.encode("utf-8", "ignore")

        # target span (in bytes)
        value = row[col_name]
        start_idx, end_idx = _find_value_span_in_row(serialized, col_name, value)

        # ids/mask
        ids, attn = _bytes_to_ids(b, cfg.src_max_len, cfg.pad_token, cfg.remap_255_to)
        valid_len = sum(attn)

        if cfg.windowed_loss:
            win_start, win_end = start_idx, min(end_idx, valid_len - 1)
        else:
            win_start, win_end = 0, int(valid_len - 1)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "start_idx": torch.tensor(start_idx, dtype=torch.long),
            "end_idx": torch.tensor(min(end_idx, valid_len - 1), dtype=torch.long),
            "win_start": torch.tensor(win_start, dtype=torch.long),
            "win_end": torch.tensor(win_end, dtype=torch.long),
            "answer_text": str(value),
        }

    def __iter__(self):
        while True:
            yield self._build_index_example()

def make_loader(cfg, split: str, batch_size: int, dataframe: Optional[pd.DataFrame]=None) -> DataLoader:
    ds = TableIndexQAStream(cfg, split, dataframe=dataframe)
    num_workers = cfg.num_workers_train if split == "train" else cfg.num_workers_val
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
                      pin_memory=(torch.cuda.is_available() and num_workers > 0))
