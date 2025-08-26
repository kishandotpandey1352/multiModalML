# data_modules/tabqa_synth.py
from typing import Iterator, Dict, Any, List, Tuple, Optional
import random, re
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _infer_type(v: Any) -> str:
    if pd.isna(v): return "NULL"
    if isinstance(v, (int, float)): return "NUM"
    s = str(v)
    if DATE_RE.match(s): return "DATE"
    if isinstance(v, str) and len(s) <= 32: return "CAT"
    return "TXT"

def _serialize_row_for_tableqa(
    row: Dict[str, Any],
    add_schema: bool,
    decompose_date: bool
) -> Tuple[str, Dict[str, Tuple[int,int]]]:
    """
    Returns: (serialized_string, map: column_name -> (byte_start, byte_end) of the *value token*)
    We make the value token format distinct per column to disambiguate duplicates.
    """
    keys = list(row.keys())
    types = {k: _infer_type(row[k]) for k in keys}
    parts: List[str] = []
    if add_schema:
        parts.append("SCHEMA: " + "; ".join(f"{k}[{types[k]}]" for k in keys))
    vals: List[str] = []
    # We'll build in text first, then compute byte offsets on utf-8
    for k in keys:
        t, v = types[k], row[k]
        if pd.isna(v):
            vals.append(f"{k}=<NULL>")
            continue
        if t == "DATE" and decompose_date:
            s = str(v)
            try:
                y, m, d = s.split("-")
                vals.append(f"{k}=<DATE:{s};year:{y};month:{m};day:{d}>")
            except Exception:
                vals.append(f"{k}=<DATE:{s}>")
        elif t == "NUM":
            vals.append(f"{k}=<NUM:{v}>")
        elif t == "CAT":
            vals.append(f"{k}=<CAT:{v}>")
        elif t == "TXT":
            s = str(v).replace("\n", " ").strip()
            vals.append(f"{k}=<TXT:{s}>")
        else:
            vals.append(f"{k}=<{v}>")
    row_line = "ROW: " + " | ".join(vals)
    parts.append(row_line)
    serial = "\n".join(parts)

    # Build a reverse index: column_name -> span over bytes
    b = serial.encode("utf-8", errors="ignore")
    spans: Dict[str, Tuple[int,int]] = {}
    start_idx = 0
    # compute by searching for "k=<TAG:val>" inside row_line reliably
    offset0 = (parts[0] + "\n").encode("utf-8", "ignore") if add_schema else b""  # schema prefix len in bytes
    base = len(offset0)
    for tok in vals:
        # token takes the form "col=<TAG:...>" or "col=<NULL>"
        # find its byte start within row_line
        sub = tok.encode("utf-8", "ignore")
        i = b.find(sub, base)   # search from after schema
        if i >= 0:
            j = i + len(sub) - 1
            # Extract col name before "=", safe for first token char until '='
            col_name = tok.split("=", 1)[0]
            spans[col_name] = (i, j)
            base = j  # next search after this
    return serial, spans

def _bytes_to_ids(b: bytes, max_len: int, pad: int, remap255: int) -> Tuple[List[int], List[int]]:
    ids = list(b)
    ids = [remap255 if t == pad else t for t in ids]  # keep PAD=255 unique
    ids = ids[:max_len]
    attn = [1] * len(ids)
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
        attn += [0] * (max_len - len(attn))
    return ids, attn

def _looks_like_id(col: str) -> bool:
    name = col.lower()
    return any(x in name for x in ["id", "uuid", "guid", "hash"])

def _choose_example(df: pd.DataFrame, cfg) -> Optional[Dict[str, Any]]:
    if df.empty: return None
    # Pick a row
    ridx = random.randrange(len(df))
    row = df.iloc[ridx].to_dict()

    # Candidate columns
    cols = list(df.columns)
    selector_pool = [c for c in (cfg.selector_cols or cols) if c in cols]
    target_pool   = [c for c in (cfg.target_cols   or cols) if c in cols]

    # Avoid targeting id-like cols
    if cfg.avoid_id_like:
        target_pool = [c for c in target_pool if not _looks_like_id(c)]

    if not selector_pool or not target_pool:
        return None

    # Pick target column different from selectors
    random.shuffle(target_pool)
    target_col = target_pool[0]

    # Build selectors (1..max) that uniquely match this row (greedy)
    random.shuffle(selector_pool)
    sel_list: List[str] = []
    for c in selector_pool:
        if c == target_col: continue
        sel_list.append(c)
        if len(sel_list) >= cfg.min_selectors:
            # Check uniqueness
            mask = pd.Series([True] * len(df))
            for s in sel_list:
                mask &= (df[s].astype(str).fillna("N/A") == str(row[s]))
            if mask.sum() == 1 or len(sel_list) >= cfg.max_selectors:
                break
    # Final uniqueness check; if not unique, skip
    mask = pd.Series([True] * len(df))
    for s in sel_list:
        mask &= (df[s].astype(str).fillna("N/A") == str(row[s]))
    if mask.sum() != 1:
        # ---- Fallback: don't stall, just use the originally picked row ----
        # Build a simple query using whatever selectors we have (or at least one)
        if not sel_list:
            # pick one non-target selector to phrase the question
            cand = [c for c in selector_pool if c != target_col]
            sel_list = cand[:1] if cand else []
        conds = " and ".join([f"{s} = {row[s]}" for s in sel_list]) if sel_list else "the selected row"
        query = f"What is the {target_col} for {conds}?"
        
        return {"query": query, "row": row, "target_col": target_col}
        
class TableQASynthStream(IterableDataset):
    """
    Generates (query + table) -> span labels completely in memory from a CSV or DataFrame.
    No downloads; no HF datasets dependency.
    """
    def __init__(self, cfg, split: str, dataframe: Optional[pd.DataFrame] = None):
        super().__init__()
        self.cfg = cfg
        self.split = split
        if dataframe is not None:
            self.df = dataframe.copy()
        elif cfg.csv_path:
            self.df = pd.read_csv(cfg.csv_path, nrows=cfg.max_rows_per_table) if cfg.use_header else \
                      pd.read_csv(cfg.csv_path, header=None, nrows=cfg.max_rows_per_table)
        else:
            raise ValueError("Provide cfg.csv_path or pass a pandas DataFrame.")
        # simple split by row index
        n = len(self.df)
        cut = int((1.0 - cfg.val_ratio) * n)
        self.df = self.df.iloc[:cut] if split == "train" else self.df.iloc[cut:]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = random.Random(self.cfg.seed + (0 if self.split=="train" else 1))
        while True:  # infinite stream; DataLoader controls steps/epochs
            ex = _choose_example(self.df, self.cfg)
            if ex is None:
                continue
            # Serialize table (single row is enough here; for robustness, you could include a small neighborhood)
            serial_row, span_map = _serialize_row_for_tableqa(
                ex["row"], self.cfg.add_schema_header, self.cfg.include_date_decompose
            )
            # Prepend the query
            full = f"Q: {ex['query']}\n{serial_row}"
            # Compute byte span of the target *value token* within the full string:
            # We search for the column token inside `serial_row` first using span_map, then offset by length of "Q: ...\n"
            value_span = span_map.get(ex["target_col"])
            if value_span is None:
                continue
            q_bytes = (f"Q: {ex['query']}\n").encode("utf-8", "ignore")
            start = len(q_bytes) + value_span[0]
            end   = len(q_bytes) + value_span[1]

            b = full.encode("utf-8", "ignore")
            ids, attn = _bytes_to_ids(b, self.cfg.src_max_len, self.cfg.pad_token, self.cfg.remap_255_to)

            # clamp spans to max_len
            if start >= len(b) or end >= len(b):
                continue
            if start >= self.cfg.src_max_len or end >= self.cfg.src_max_len:
                continue

            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "start_idx": torch.tensor(start, dtype=torch.long),
                "end_idx": torch.tensor(end, dtype=torch.long),
                "answer_text": str(ex["row"][ex["target_col"]]),
            }

def _collate_tabqa(batch):
    """
    Collate a list[dict] -> dict of stacked tensors/lists.
    Keeps non-tensor fields (e.g., 'answer_text') as lists.
    """
    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out

def make_loader(cfg, split: str, batch_size: int, dataframe=None):
    ds = TableQASynthStream(cfg, split, dataframe=dataframe)

    using_cuda = torch.cuda.is_available()
    pin = using_cuda
    num_workers = getattr(cfg, "num_workers_train", 4) if split == "train" else getattr(cfg, "num_workers_val", 2)
    num_workers = max(0, int(num_workers))

    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin, collate_fn=_collate_tabqa, drop_last=False)
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=getattr(cfg, "prefetch_factor", 2))
    return DataLoader(ds, **kwargs)
