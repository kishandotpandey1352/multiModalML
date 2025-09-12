
# data_modules/audio_hf.py
# -----------------------------------------------------------------------------
# Streaming Audio → Byte-IDs loader for ESC-50 (and similar HF datasets)
# Key features:
#  - **Resampling** to cfg.sample_rate BEFORE cropping, so `max_secs` is real time.
#  - **Optional byte thinning** via cfg.byte_stride (e.g., 2) after int16 conversion,
#    letting you fit longer windows under cfg.src_max_len without raising encoder size.
#  - **Attention-friendly outputs**: returns input_ids and attention_mask (pad=255).
#  - **Lightweight streaming**: safe defaults for workers/pinning/prefetching.
#  - Robust split/repo fallbacks for ESC-50 mirrors.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Iterator, Dict, Any, Optional
import io, itertools, random, numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

PAD_TOKEN = 255  # default pad token (kept for mask building)


# --------------------- helpers: audio↔bytes ↔ ids ---------------------
def _float32_to_int16_bytes(wave: torch.Tensor) -> bytes:
    """
    Convert mono float32 waveform in [-1, 1] to little-endian int16 bytes.
    """
    wave16 = (wave.clamp(-1, 1) * 32767.0).to(torch.int16).cpu().numpy()
    return wave16.tobytes(order="C")


def _bytes_to_ids(b: bytes, max_len: int, pad: int, remap255: int) -> tuple[list[int], list[int]]:
    """
    Map raw bytes to token IDs in [0..255], remapping 255 to remap255 so that
    255 remains available as PAD. Build a matching attention mask.
      - ids length == max_len (padded if short, truncated if long)
      - attn length == max_len (1 where real, 0 where pad)
    """
    if not b:
        ids = []
    else:
        ids = [remap255 if t == pad else t for t in b]
    ids = ids[:max_len]
    attn = [1] * len(ids)
    if len(ids) < max_len:
        padlen = max_len - len(ids)
        ids  += [pad] * padlen
        attn += [0]   * padlen
    return ids, attn


# ------------------------ HF streaming utilities ------------------------
def _try_repo_stream(repo_id: str, split: str):
    return load_dataset(repo_id, split=split, streaming=True)


def _select_esc50_stream(cfg, split_key: str):
    """
    Choose a streaming ESC-50 source with helpful fallbacks.
    Order:
      1) cfg.preferred_repos (list) if provided
      2) cfg.dataset_name (single) if provided
      3) common mirrors: ["ashraq/esc50", "mskov/ESC50"]
    Splits:
      - train → ["train"]
      - val   → ["validation", "valid", "val", "test"]
      - else  → [split_key, "validation", "test"]
    """
    repos = list(getattr(cfg, "preferred_repos", []) or [])
    dname = getattr(cfg, "dataset_name", None)
    if dname and dname not in repos:
        repos.append(dname)
    if not repos:
        repos = ["ashraq/esc50", "mskov/ESC50"]

    if split_key == getattr(cfg, "split_train", "train"):
        split_candidates = ["train"]
    elif split_key == getattr(cfg, "split_val", "validation"):
        split_candidates = ["validation", "valid", "val", "test"]
    else:
        split_candidates = [split_key, "validation", "test"]

    last_err = None
    for repo in repos:
        for sp in split_candidates:
            try:
                ds = _try_repo_stream(repo, sp)
                print(f"[INFO] ESC-50 source selected for split='{split_key}': repo:{repo}/{sp}")
                return ds
            except Exception as e:
                last_err = e
                print(f"[WARN] repo stream failed: {repo}/{sp} -> {e}")
    raise RuntimeError(f"All repo loading strategies failed for '{split_key}'. Last error: {last_err}")


def _resolve_audio_to_wave(ex: Dict[str, Any], audio_field: str, parquet_repo: Optional[str]):
    """
    Return (mono_float32_wave, sr).
    Accepts typical HF audio column dicts or {'bytes'} / {'path'} / {'file'} / raw array.
    """
    import torchaudio, fsspec

    # datasets 'Audio' column
    if audio_field in ex and isinstance(ex[audio_field], dict) and \
       "array" in ex[audio_field] and "sampling_rate" in ex[audio_field]:
        wav = torch.tensor(ex[audio_field]["array"], dtype=torch.float32)
        sr = int(ex[audio_field]["sampling_rate"])
        return wav, sr

    # embedded bytes
    if audio_field in ex and isinstance(ex[audio_field], dict) and "bytes" in ex[audio_field]:
        raw = ex[audio_field]["bytes"]
        raw = bytes(raw) if isinstance(raw, list) else raw
        bio = io.BytesIO(raw)
        wav, sr = torchaudio.load(bio)
        wav = wav.mean(dim=0) if wav.dim() == 2 else wav.squeeze(0)
        return wav.to(torch.float32), int(sr)

    # file path directly
    if "file" in ex:
        wav, sr = torchaudio.load(ex["file"])
        wav = wav.mean(dim=0) if wav.dim() == 2 else wav.squeeze(0)
        return wav.to(torch.float32), int(sr)

    # 'path' dict (optionally relative to HF dataset files)
    if audio_field in ex and isinstance(ex[audio_field], dict) and "path" in ex[audio_field]:
        path = ex[audio_field]["path"]
        if not (str(path).startswith(("http://", "https://", "hf://"))) and parquet_repo:
            path = f"hf://datasets/{parquet_repo}/{path}"
        with fsspec.open(path, "rb") as f:
            data = f.read()
        bio = io.BytesIO(data)
        wav, sr = torchaudio.load(bio)
        wav = wav.mean(dim=0) if wav.dim() == 2 else wav.squeeze(0)
        return wav.to(torch.float32), int(sr)

    # raw numeric array
    if audio_field in ex and isinstance(ex[audio_field], (list, tuple, np.ndarray)):
        wav = torch.tensor(ex[audio_field], dtype=torch.float32)
        sr  = int(ex.get("sampling_rate", 44100))
        return wav, sr

    raise KeyError(f"Unsupported audio schema. Keys: {list(ex.keys())[:10]}")


# ------------------------ Streaming dataset class ------------------------
class AudioBytesStream(IterableDataset):
    def __init__(self, cfg, split: str):
        self.cfg = cfg
        self.split = split
        self.ds = _select_esc50_stream(cfg, split)
        self._class_to_id = {}

    def _extract_label(self, ex: Dict[str, Any]) -> int:
        lf = getattr(self.cfg, "label_field", None)
        if lf and lf in ex:
            try:
                return int(ex[lf])
            except Exception:
                pass
        for k in ("target", "label"):
            if k in ex:
                try:
                    return int(ex[k])
                except Exception:
                    continue
        if "category" in ex:
            cat = str(ex["category"])
            if cat not in self._class_to_id:
                self._class_to_id[cat] = len(self._class_to_id)
            return self._class_to_id[cat]
        raise KeyError("No label found (checked cfg.label_field, 'target', 'label', 'category').")

    def __iter__(self):
        import torchaudio
        import torchaudio.functional as AF
        import torch.nn.functional as F

        # ----- config -----
        target_sr   = int(getattr(self.cfg, "sample_rate", 44100))  # RESAMPLE target
        secs        = float(getattr(self.cfg, "max_secs", 5.0))
        max_len     = int(getattr(self.cfg, "src_max_len", 1024))
        pad_tok     = int(getattr(self.cfg, "pad_token", PAD_TOKEN))
        remap255    = int(getattr(self.cfg, "remap_255_to", 254))
        byte_stride = int(getattr(self.cfg, "byte_stride", 1))      # 1 = no thinning
        audio_field  = getattr(self.cfg, "audio_field", "audio")
        parquet_repo = getattr(self.cfg, "parquet_repo", None)

        # stream cap
        it = self.ds
        take = None
        if self.split == getattr(self.cfg, "split_val", "validation"):
            take = getattr(self.cfg, "val_stream_take", None)
        if take is None:
            take = getattr(self.cfg, "stream_take", None)
        if take:
            it = itertools.islice(it, int(take))

        is_train = (self.split == getattr(self.cfg, "split_train", "train"))
        rand_crop = bool(getattr(self.cfg, "random_offset_crop", True))
        use_augs  = bool(getattr(self.cfg, "use_augs", False))

        for ex in it:
            wav, sr = _resolve_audio_to_wave(ex, audio_field, parquet_repo)

            # ensure mono float32
            if wav.dim() > 1:
                wav = wav.mean(dim=-1)
            wav = wav.to(torch.float32)

            # ----- RESAMPLE to target_sr (crucial so secs→samples is stable) -----
            if sr != target_sr:
                wav = AF.resample(wav, orig_freq=sr, new_freq=target_sr)
                sr = target_sr

            # duration crop/pad at target_sr
            num_target = int(sr * secs)
            if wav.numel() >= num_target:
                start = random.randint(0, max(0, wav.numel() - num_target)) if (is_train and rand_crop) else 0
                wav = wav[start:start + num_target]
            else:
                wav = F.pad(wav, (0, num_target - wav.numel()))

            # simple and safe augments
            if is_train and use_augs:
                if random.random() < 0.25:
                    wav = (wav + torch.randn_like(wav) * 0.005).clamp(-1, 1)  # light noise
                if random.random() < 0.25:
                    wav = (wav * random.uniform(0.8, 1.2)).clamp(-1, 1)      # gain

            # to int16 bytes
            b = _float32_to_int16_bytes(wav)

            # optional byte thinning to fit longer windows under token cap
            if byte_stride > 1:
                b = b[::byte_stride]

            # build token ids + attention
            ids, attn = _bytes_to_ids(b, max_len=max_len, pad=pad_tok, remap255=remap255)

            # safety shape check
            if len(ids) != max_len or len(attn) != max_len:
                ids  = (ids + [pad_tok] * max_len)[:max_len]
                attn = (attn + [0] * max_len)[:max_len]

            y = self._extract_label(ex)

            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "label": torch.tensor(int(y), dtype=torch.long),
            }


# ----------------------------- DataLoader factory -----------------------------
def make_loader(cfg, split: str, batch_size: int) -> DataLoader:
    ds = AudioBytesStream(cfg, split)
    # Streaming-friendly loader defaults
    nw = min(1, int(getattr(cfg, "num_workers", 0)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=bool(getattr(cfg, "pin_memory", False)),
        persistent_workers=False,
        prefetch_factor=1 if nw > 0 else None,
    )
