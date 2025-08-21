# data_modules/tab_hf.py
from typing import Iterator, Dict, Any, List
import os
import re
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

# ---- Robust exception imports across datasets versions ----
try:
    from datasets.exceptions import DatasetNotFoundError, DatasetGenerationError
except Exception:
    try:
        from datasets.builder import DatasetGenerationError  # type: ignore
    except Exception:
        class DatasetGenerationError(Exception):
            ...
    class DatasetNotFoundError(Exception):
        ...

# Optional verification mode (varies by version)
try:
    from datasets import VerificationMode
    _NO_CHECKS = VerificationMode.NO_CHECKS
except Exception:
    _NO_CHECKS = "no_checks"

from huggingface_hub import hf_hub_download
import pandas as pd

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Common alias map so slugs "just work"
ALIASES = {
    "adult": "scikit-learn/adult-census-income",
    "bank_marketing": "bank-marketing",
}

def _bytes_to_ids(b: bytes, max_len: int, pad: int, remap255: int) -> tuple[list[int], list[int]]:
    ids = list(b)
    # keep PAD=255 unique by remapping any content 255 to remap255 (e.g., 254)
    ids = [remap255 if t == pad else t for t in ids]
    ids = ids[:max_len]
    attn = [1] * len(ids)
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
        attn += [0] * (max_len - len(attn))
    return ids, attn

def _infer_type(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return "NUM"
    s = str(v)
    if DATE_RE.match(s):
        return "DATE"
    if isinstance(v, str) and len(s) <= 32:
        return "CAT"
    return "TXT"

def _serialize_row(row: Dict[str, Any], add_schema: bool, decompose_date: bool) -> str:
    keys = list(row.keys())
    types = {k: _infer_type(row[k]) for k in keys}
    parts = []
    if add_schema:
        parts.append("SCHEMA: " + "; ".join(f"{k}[{types[k]}]" for k in keys))
    vals = []
    for k in keys:
        t, v = types[k], row[k]
        if v is None:
            vals.append(f"{k}=<NULL>")
            continue
        if t == "DATE" and decompose_date:
            try:
                y, m, d = str(v).split("-")
                vals.append(f"{k}=<DATE:{v};year:{y};month:{m};day:{d}>")
            except Exception:
                vals.append(f"{k}=<DATE:{v}>")
        elif t == "NUM":
            vals.append(f"{k}=<NUM:{v}>")
        elif t == "CAT":
            vals.append(f"{k}=<CAT:{v}>")
        elif t == "TXT":
            s = str(v).replace("\n", " ").strip()
            vals.append(f"{k}=<TXT:{s}>")
        else:
            vals.append(f"{k}=<{v}>")
    parts.append("ROW: " + " | ".join(vals))
    return "\n".join(parts)

def _slice_indices(split_str: str, n: int) -> tuple[int, int]:
    # supports "train", "train[:90%]", "train[90%:]", "train[10%:95%]"
    if "[" not in split_str:
        return 0, n
    inside = split_str.split("[", 1)[1].rstrip("]")

    def to_idx(tok: str) -> int:
        tok = tok.strip()
        if tok.endswith("%"):
            return int(round(float(tok[:-1]) / 100.0 * n))
        return int(tok) if tok else 0

    if ":" in inside:
        left, right = inside.split(":", 1)
        i0 = to_idx(left) if left else 0
        i1 = to_idx(right) if right else n
    else:
        i0, i1 = 0, to_idx(inside)
    i0 = max(0, min(i0, n))
    i1 = max(0, min(i1, n))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1

# ---------- Manual Adult CSV helpers ----------
def _adult_csv_path() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return hf_hub_download(
        repo_id="scikit-learn/adult-census-income",
        filename="adult.csv",
        repo_type="dataset",
        token=token,
    )

def _manual_adult_columns() -> List[str]:
    csv_path = _adult_csv_path()
    df = pd.read_csv(csv_path, nrows=1)
    return list(df.columns)

def _manual_adult_rows(split: str) -> Iterator[Dict[str, Any]]:
    csv_path = _adult_csv_path()
    df = pd.read_csv(csv_path)  # avoids datasets' pandas kw mismatches
    n = len(df)
    i0, i1 = _slice_indices(split, n)
    view = df.iloc[i0:i1]
    for _, row in view.iterrows():
        yield row.to_dict()

class TabBytesStream(IterableDataset):
    """
    Streams HF tabular dataset rows as UTF-8 bytes for the shared encoder.
    Produces: input_ids [L], attention_mask [L], label (0/1 float).
    """
    def __init__(self, cfg, split: str):
        self.cfg = cfg
        self.split = split
        raw_name = getattr(cfg, "dataset_name", "scikit-learn/adult-census-income")
        name = ALIASES.get(raw_name, raw_name)
        self._manual = False
        self._name = name  # keep for debugging

        # Try native datasets loader
        try:
            self.ds = load_dataset(
                name,
                name=getattr(cfg, "dataset_config", None),
                split=split,
                streaming=getattr(cfg, "use_hf_streaming", False),
                verification_mode=_NO_CHECKS if getattr(cfg, "verification_no_checks", True) else None,
            )
        except (DatasetNotFoundError, DatasetGenerationError, TypeError, ValueError):
            # Fallback for Adult CSV (pandas<->datasets kw mismatch)
            if name == "scikit-learn/adult-census-income":
                print("[tab_hf] Falling back to manual CSV loader for scikit-learn/adult-census-income.")
                self.ds = None
                self._manual = True
            else:
                raise

        # ---- Sanity check label column on RAW data (NOT encoded batches) ----
        lbl = self.cfg.label_column
        if self._manual:
            cols = _manual_adult_columns()
            if lbl not in cols:
                raise RuntimeError(f"[tab_hf] Label column '{lbl}' not found in CSV. Available: {', '.join(cols)}")
        else:
            # Peek a single raw example
            try:
                raw_ex = next(iter(self.ds))
            except StopIteration:
                raise RuntimeError("[tab_hf] Dataset split is empty after loading.")
            if lbl not in raw_ex:
                cols = ", ".join(sorted(raw_ex.keys()))
                raise RuntimeError(f"[tab_hf] Label column '{lbl}' not found. Available: {cols}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._manual:
            iterator = _manual_adult_rows(self.split)
        else:
            iterator = iter(self.ds)

        label_col = self.cfg.label_column
        pos_label = None if self.cfg.positive_label is None else str(self.cfg.positive_label)

        for ex in iterator:
            if label_col not in ex:
                continue

            # Map to binary float label
            if pos_label is None:
                # Expect numeric {0,1}
                try:
                    y = int(ex[label_col])
                except Exception:
                    continue
            else:
                y = 1 if str(ex[label_col]) == pos_label else 0

            feat = {k: v for k, v in ex.items() if k != label_col}
            s = _serialize_row(feat, self.cfg.add_schema_header, self.cfg.include_date_decompose)
            b = s.encode("utf-8", errors="ignore")
            ids, attn = _bytes_to_ids(b, self.cfg.src_max_len, self.cfg.pad_token, self.cfg.remap_255_to)

            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "label": torch.tensor(float(y), dtype=torch.float32),
            }

def make_loader(cfg, split: str, batch_size: int) -> DataLoader:
    ds = TabBytesStream(cfg, split)
    return DataLoader(ds, batch_size=batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
