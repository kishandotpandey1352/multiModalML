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

            # Serialize a single row (schema header optional)
            serial_row, span_map = _serialize_row_for_tableqa(
                ex["row"], self.cfg.add_schema_header, self.cfg.include_date_decompose
            )
            # Build the full prompt (question + row)
            full = f"Q: {ex['query']}\n{serial_row}"

            # Token-level span (covers "col=<TAG:val>" or "col=<NULL>")
            tok_span = span_map.get(ex["target_col"])
            if tok_span is None:
                continue

            b = full.encode("utf-8", "ignore")
            q_bytes = (f"Q: {ex['query']}\n").encode("utf-8", "ignore")
            q_off = len(q_bytes)

            # --- derive value-only span inside the token ---
            ti, tj = tok_span  # token byte [start, end] within serial_row inside `full` after q_off
            ti_full, tj_full = q_off + ti, q_off + tj
            token_bytes = b[ti_full:tj_full+1]

            # Cases:
            #   "<NULL>" -> value = "NULL"
            #   "<TAG:VALUE>" (DATE/NUM/CAT/TXT) -> value between first ":" and next ";" or ">"
            val_start_off = None
            val_end_off = None
            if b"<NULL>" in token_bytes:
                pos = token_bytes.find(b"NULL")
                if pos >= 0:
                    val_start_off = ti_full + pos
                    val_end_off   = val_start_off + len(b"NULL") - 1
            else:
                # find the first ':' after '<'
                lt = token_bytes.find(b"<")
                colon = token_bytes.find(b":", lt + 1 if lt >= 0 else 0)
                if colon >= 0:
                    # value ends at next ';' (for DATE extra fields) or '>'
                    semi = token_bytes.find(b";", colon + 1)
                    gt   = token_bytes.find(b">", colon + 1)
                    ends = [x for x in [semi, gt] if x >= 0]
                    if ends:
                        stop = min(ends)
                        val_start_off = ti_full + colon + 1
                        val_end_off   = ti_full + stop - 1

            # Fallback: if we failed to parse value span, skip this example
            if val_start_off is None or val_end_off is None:
                continue

            # Byte ids + mask
            ids, attn = _bytes_to_ids(b, self.cfg.src_max_len, self.cfg.pad_token, self.cfg.remap_255_to)

            # Clamp spans to max_len and real bytes
            if val_start_off >= len(b) or val_end_off >= len(b):
                continue
            if val_start_off >= self.cfg.src_max_len or val_end_off >= self.cfg.src_max_len:
                continue
            
            value_text = b[val_start_off:val_end_off+1].decode("utf-8","ignore").strip()
            ids, attn = _bytes_to_ids(b, self.cfg.src_max_len, self.cfg.pad_token, self.cfg.remap_255_to)

            yield {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                    "start_idx": torch.tensor(val_start_off, dtype=torch.long),
                    "end_idx": torch.tensor(val_end_off, dtype=torch.long),
                    "win_start": torch.tensor(ti_full, dtype=torch.long),   # <--- NEW
                    "win_end":   torch.tensor(tj_full, dtype=torch.long),   # <--- NEW
                    "answer_text": value_text,
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
