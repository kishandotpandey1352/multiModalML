# data_modules/img_hf_fewshot.py
from typing import Iterator, Dict, Any
from io import BytesIO
import random
import itertools
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

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
    ids = [remap255 if t == pad else t for t in ids]  # keep 255 as PAD
    ids = ids[:max_len]
    attn = [1]*len(ids)
    if len(ids) < max_len:
        ids += [pad]*(max_len - len(ids))
        attn += [0]*(max_len - len(attn))
    return ids, attn

class ImageBytesStreamFewShot(IterableDataset):
    """
    Few-shot train stream: yields at most K examples per class.
    For non-train splits, behaves like full stream.
    """
    def __init__(self, cfg, split: str, fewshot_per_class: int | None = None, max_total: int | None = None):
        self.cfg = cfg
        self.split = split
        self.fewshot_per_class = fewshot_per_class if split == cfg.split_train else None
        self.max_total = max_total if split == cfg.split_train else None

        self.ds = load_dataset(
            cfg.dataset_name,
            name=cfg.dataset_config,
            split=split,
            streaming=cfg.use_hf_streaming,
            verification_mode=_NO_CHECKS if getattr(cfg, "verification_no_checks", True) else None,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import PIL.Image as Image
        from PIL import ImageOps

        resize = self.cfg.resize
        img_field_candidates = getattr(self.cfg, "img_field_candidates", ("img","image"))
        label_field = self.cfg.label_field

        # Class-balanced counters (train few-shot only)
        k = self.fewshot_per_class
        per_class = [0] * getattr(self.cfg, "num_classes", 10)
        taken_total = 0
        budget_total = None
        if k is not None:
            budget_total = k * len(per_class)
            if self.max_total is not None:
                budget_total = min(budget_total, self.max_total)

        # Optional overall cap for streaming
        stream_iter = self.ds
        if getattr(self.cfg, "stream_take", None):
            stream_iter = itertools.islice(self.ds, int(self.cfg.stream_take))

        for ex in stream_iter:
            # stop if reached few-shot quota
            if k is not None and budget_total is not None and taken_total >= budget_total:
                break

            # locate image
            img = None
            for f in img_field_candidates:
                if f in ex:
                    img = ex[f]; break
            if img is None: 
                continue

            # PIL image
            if hasattr(img, "convert"):
                im = img.convert("RGB")
            else:
                im = Image.fromarray(img).convert("RGB")

            # augment only on train split
            if self.split == self.cfg.split_train:
                # random horizontal flip
                if random.random() < 0.5:
                    im = im.transpose(method=Image.FLIP_LEFT_RIGHT)
                # random pad + random crop (≈ 12.5% padding)
                w, h = im.size
                pad = max(1, int(0.125 * min(w, h)))
                im = ImageOps.expand(im, border=pad, fill=0)
                nx, ny = w + 2*pad, h + 2*pad
                x0 = random.randint(0, nx - w)
                y0 = random.randint(0, ny - h)
                im = im.crop((x0, y0, x0 + w, y0 + h))

            # resize (shorter side)
            if resize and resize > 0:
                w, h = im.size
                if min(w,h) != resize:
                    if w < h:
                        nw = resize; nh = int(h * (resize / w))
                    else:
                        nh = resize; nw = int(w * (resize / h))
                    im = im.resize((nw, nh), Image.BICUBIC)

            # to bytes: raw RGB or PNG
            if getattr(self.cfg, "to_png_bytes", False):
                b = _pil_to_png_bytes(im)
            else:
                b = im.tobytes()  # row-major RGB

            y = int(ex[label_field])
            # enforce per-class quota for few-shot
            if k is not None:
                if y < 0 or y >= len(per_class):  # safety
                    continue
                if per_class[y] >= k:
                    continue
                per_class[y] += 1
                taken_total += 1

            ids, attn = _bytes_to_ids(b, self.cfg.src_max_len, self.cfg.pad_token, self.cfg.remap_255_to)

            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "label": torch.tensor(y, dtype=torch.long),
            }

def make_loader_fewshot(cfg, split: str, batch_size: int, fewshot_per_class: int | None = None, max_total: int | None = None) -> DataLoader:
    ds = ImageBytesStreamFewShot(cfg, split, fewshot_per_class=fewshot_per_class, max_total=max_total)
    return DataLoader(ds, batch_size=batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
