# data_modules/audio_hf.py  (only the loader parts shown here)

from __future__ import annotations
from typing import Iterator, Dict, Any, Optional, List
import io, itertools, random, numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

# ---------- helpers: audio bytes/id conversion (keep yours if already present) ----------
def _float32_to_int16_bytes(wave: torch.Tensor) -> bytes:
    wave16 = (wave.clamp(-1, 1) * 32767.0).to(torch.int16).cpu().numpy()
    return wave16.tobytes(order="C")

def _bytes_to_ids(b: bytes, max_len: int, pad: int, remap255: int) -> tuple[list[int], list[int]]:
    ids = [remap255 if t == pad else t for t in b]
    ids = ids[:max_len]
    attn = [1] * len(ids)
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
        attn += [0] * (max_len - len(attn))
    return ids, attn

# ---------- NEW: robust source resolver ----------
def _try_parquet_stream(parquet_repo: str, split_glob: Optional[str]):
    """Try hf parquet streaming: load_dataset('parquet', data_files='hf://datasets/<repo>/<glob>', streaming=True)."""
    if not parquet_repo or not split_glob:
        raise ValueError("parquet_repo or split_glob missing")
    data_files = f"hf://datasets/{parquet_repo}/{split_glob}"
    return load_dataset("parquet", data_files=data_files, split="train", streaming=True)

def _try_repo_stream(repo_id: str, split: str):
    return load_dataset(repo_id, split=split, streaming=True)

def _load_any_esc50_stream(cfg, split_key: str):
    """
    Try, in order:
      1) preferred repos from config (e.g., "ashraq/esc50", "mskov/ESC50")
      2) (optional) cfg.dataset_name if present
    For each repo, try a small set of common split names:
      - for 'train'  -> ['train']
      - for 'val'    -> ['validation', 'valid', 'val', 'test']
      - for 'test'   -> ['test', 'validation']
    Returns: (IterableDataset, "repo:<id>/<split>")
    """
    # Build repo trial list
    repos = list(getattr(cfg, "preferred_repos", []) or [])
    dname = getattr(cfg, "dataset_name", None)
    if dname and dname not in repos:
        repos.append(dname)

    # Fallback to these if not provided
    if not repos:
        repos = ["ashraq/esc50", "mskov/ESC50"]

    # Split candidates
    if split_key == getattr(cfg, "split_train", "train"):
        split_candidates = ["train"]
    elif split_key == getattr(cfg, "split_val", "validation"):
        split_candidates = ["validation", "valid", "val", "test"]
    else:
        # generic guess
        split_candidates = [split_key, "validation", "test"]

    last_err = None
    for repo in repos:
        for sp in split_candidates:
            try:
                ds = _try_repo_stream(repo, sp)
                return ds, f"repo:{repo}/{sp}"
            except Exception as e:
                last_err = e
                print(f"[WARN] repo stream failed: {repo}/{sp} -> {e}")

    raise RuntimeError(
        f"All repo loading strategies failed for '{split_key}'. Last error: {last_err}"
    )

# ---------- your existing resolver for 'audio'/{array,bytes,path} can remain ----------
def _resolve_audio_to_wave(ex: Dict[str, Any], audio_field: str, target_sr: int, parquet_repo: Optional[str]) -> tuple[torch.Tensor, int]:
    import torchaudio, fsspec

    if audio_field in ex and isinstance(ex[audio_field], dict) and "array" in ex[audio_field] and "sampling_rate" in ex[audio_field]:
        wav = torch.tensor(ex[audio_field]["array"], dtype=torch.float32)
        sr = int(ex[audio_field]["sampling_rate"])
        return wav, sr

    if audio_field in ex and isinstance(ex[audio_field], dict) and "bytes" in ex[audio_field]:
        raw = ex[audio_field]["bytes"]
        raw = bytes(raw) if isinstance(raw, list) else raw
        bio = io.BytesIO(raw)
        wav, sr = torchaudio.load(bio)
        wav = wav.mean(dim=0) if wav.dim() == 2 else wav.squeeze(0)
        return wav.to(torch.float32), int(sr)

    if "file" in ex:
        # direct path (common in ESC-50 mirrors)
        import torchaudio
        wav, sr = torchaudio.load(ex["file"])
        wav = wav.mean(dim=0) if wav.dim() == 2 else wav.squeeze(0)
        return wav.to(torch.float32), int(sr)

    if audio_field in ex and isinstance(ex[audio_field], dict) and "path" in ex[audio_field]:
        path = ex[audio_field]["path"]
        if not (str(path).startswith(("http://","https://","hf://"))) and parquet_repo:
            path = f"hf://datasets/{parquet_repo}/{path}"
        with fsspec.open(path, "rb") as f:
            data = f.read()
        bio = io.BytesIO(data)
        wav, sr = torchaudio.load(bio)
        wav = wav.mean(dim=0) if wav.dim() == 2 else wav.squeeze(0)
        return wav.to(torch.float32), int(sr)

    if audio_field in ex and isinstance(ex[audio_field], (list, tuple, np.ndarray)):
        wav = torch.tensor(ex[audio_field], dtype=torch.float32)
        sr = int(ex.get("sampling_rate", target_sr))
        return wav, sr

    raise KeyError(f"Unsupported audio schema. Keys: {list(ex.keys())[:10]}")

# ---------- dataset class ----------
class AudioBytesStream(IterableDataset):
    def __init__(self, cfg, split: str):
        self.cfg = cfg
        self.split = split
        # Do NOT try parquet first; it fails on your env. Go repo-first.
        self.ds, src = _load_any_esc50_stream(cfg, split)
        print(f"[INFO] ESC-50 source selected for split='{split}': {src}")
        self._class_to_id = {}

    def _extract_label(self, ex: Dict[str, Any]) -> int:
        lf = getattr(self.cfg, "label_field", None)
        if lf and lf in ex:
            try: return int(ex[lf])
            except Exception: pass
        for k in ("target", "label"):
            if k in ex:
                try: return int(ex[k])
                except Exception: continue
        if "category" in ex:
            cat = str(ex["category"])
            if cat not in self._class_to_id:
                self._class_to_id[cat] = len(self._class_to_id)
            return self._class_to_id[cat]
        raise KeyError("No label found (checked cfg.label_field, 'target', 'label', 'category').")

    def __iter__(self):
        import torchaudio
        target_sr = int(getattr(self.cfg, "sample_rate", 44100))  # not used if we skip resample
        secs      = float(getattr(self.cfg, "max_secs", 5.0))
        max_len   = int(getattr(self.cfg, "src_max_len", 1024))
        pad_tok   = int(getattr(self.cfg, "pad_token", 255))
        remap255  = int(getattr(self.cfg, "remap_255_to", 254))
        audio_field  = getattr(self.cfg, "audio_field", "audio")
        parquet_repo = getattr(self.cfg, "parquet_repo", None)

        it = self.ds
        # Cap rows per epoch. Use val_stream_take for val split, stream_take otherwise.
        take = None
        if self.split == getattr(self.cfg, "split_val", "validation"):
            take = getattr(self.cfg, "val_stream_take", None)
        if take is None:
            take = getattr(self.cfg, "stream_take", None)
        if take:
            it = itertools.islice(it, int(take))

        is_train = (self.split == getattr(self.cfg, "split_train", "train"))

        for ex in it:
            wav, sr = _resolve_audio_to_wave(ex, audio_field, target_sr, parquet_repo)
            if wav.dim() > 1:
                wav = wav.mean(dim=-1)

            num_target = int(sr * secs)
            if wav.numel() >= num_target:
                start = random.randint(0, max(0, wav.numel() - num_target)) if (is_train and getattr(self.cfg, "random_offset_crop", True)) else 0
                wav = wav[start:start+num_target]
            else:
                wav = torch.nn.functional.pad(wav, (0, num_target - wav.numel()))

            if is_train and getattr(self.cfg, "use_augs", False):
                if random.random() < 0.25:
                    wav = (wav + torch.randn_like(wav) * 0.005).clamp(-1, 1)
                if random.random() < 0.25:
                    wav = (wav * random.uniform(0.8, 1.2)).clamp(-1, 1)

            b = _float32_to_int16_bytes(wav)
            ids, attn = _bytes_to_ids(b, max_len=max_len, pad=pad_tok, remap255=remap255)
            # extra safety if something upstream changes
            if len(ids) != max_len or len(attn) != max_len:
                ids = (ids + [pad_tok] * max_len)[:max_len]
                attn = (attn + [0] * max_len)[:max_len]            
            
            y = self._extract_label(ex)

            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "label": torch.tensor(int(y), dtype=torch.long),
            }

def make_loader(cfg, split: str, batch_size: int) -> DataLoader:
    ds = AudioBytesStream(cfg, split)
    nw = min(1, int(getattr(cfg, "num_workers", 0)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=bool(getattr(cfg, "pin_memory", False)),
        persistent_workers=False,
        prefetch_factor=1 if nw > 0 else None,  # keep tiny
    )
