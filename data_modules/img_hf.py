#img_hf.py
from typing import Iterator, Dict, Any
from io import BytesIO
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from PIL import ImageOps

try:
    from datasets import VerificationMode
    _NO_CHECKS = VerificationMode.NO_CHECKS
except Exception:
    _NO_CHECKS = "no_checks"

def _pil_to_png_bytes(pil_img) -> bytes:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def _bytes_to_ids(b: bytes, max_len: int, pad: int, remap255: int) -> tuple[list[int], list[int]]:
    ids = list(b)
    # remap any 255 in content to 254 so 255 uniquely means PAD
    ids = [remap255 if t == pad else t for t in ids]
    ids = ids[:max_len]
    attn = [1]*len(ids)
    if len(ids) < max_len:
        ids += [pad]*(max_len - len(ids))
        attn += [0]*(max_len - len(attn))
    return ids, attn

class ImageBytesStream(IterableDataset):
    def __init__(self, cfg, split: str):
        self.cfg = cfg
        self.split = split
        self.ds = load_dataset(
            cfg.dataset_name,
            name=cfg.dataset_config,
            split=split,
            streaming=cfg.use_hf_streaming,
            verification_mode=_NO_CHECKS if cfg.verification_no_checks else None,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import PIL.Image as Image
        import numpy as np
        resize = self.cfg.resize
        for ex in self.ds:
            # find image field
            img = None
            for f in self.cfg.img_field_candidates:
                if f in ex:
                    img = ex[f]
                    break
            if img is None:
                continue
            if hasattr(img, "convert"):
                im = img.convert("RGB")
            else:  # sometimes comes as np array
                im = Image.fromarray(img).convert("RGB")

            if resize and resize > 0:
                # keep aspect ratio, shorter side -> resize
                w, h = im.size
                if min(w,h) != resize:
                    if w < h:
                        nw = resize; nh = int(h * (resize / w))
                    else:
                        nh = resize; nw = int(w * (resize / h))
                    im = im.resize((nw, nh), Image.BICUBIC)

            if self.cfg.to_png_bytes:
                b = _pil_to_png_bytes(im)
            else:
                # raw RGB bytes (row-major) — tends to be longer; optional
                b = im.tobytes()

            ids, attn = _bytes_to_ids(
                b, self.cfg.src_max_len, self.cfg.pad_token, self.cfg.remap_255_to
            )

            y = int(ex[self.cfg.label_field])

            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "label": torch.tensor(y, dtype=torch.long),
            }

def make_loader(cfg, split: str, batch_size: int) -> DataLoader:
    ds = ImageBytesStream(cfg, split)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
