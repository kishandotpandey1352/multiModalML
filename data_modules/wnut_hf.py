
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset

from datasets import load_dataset
try:
    from datasets import VerificationMode
    _NO_CHECKS = VerificationMode.NO_CHECKS
except Exception:
    _NO_CHECKS = "no_checks"
    
def text_to_bytes(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="ignore"))

class WNUT17HFBytes(Dataset):
    """
    HF NER dataset adapter (WNUT-17/WikiANN/others) → UTF-8 bytes.
    Automatically detects tag field 'ner_tags' or 'tags' and supports dataset configs (e.g., 'en' for WikiANN).
    """
    def __init__(
                self,
                split: str = "train",
                tag_field: str | None = None,
                dataset_config: str | None = None,   # <— NEW
                dataset_kwargs: dict | None = None,  # <— NEW (optional)
                max_len: int = 1024,
                pad_token: int = 255,
                bos_token: int = 257,
                eos_token: int = 258,
                add_bos: bool = True,
                add_eos: bool = True,
                ignore_index: int = -100,
                use_hf_streaming: bool = False,
                label_list: Optional[List[str]] = None,
                dataset_name: str = "unimelb-nlp/wikiann",
                    ):
        
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.ignore_index = ignore_index
        self.split = split
        self.use_hf_streaming = use_hf_streaming

        # Load split (with optional builder config and kwargs)
        self.ds = load_dataset(
                            dataset_name,
                            *([dataset_config] if dataset_config else []),
                            split=split,
                            streaming=use_hf_streaming,
                            verification_mode=_NO_CHECKS,
                        )

        # Determine label field & names
        self.tag_field = tag_field  # "ner_tags" (HF canonical) or "tags" (e.g., TNER repos)
        if label_list is not None:
            self.label_list = label_list
        else:
            try:
                features = None if use_hf_streaming else getattr(self.ds, "features", None)
                if features and "ner_tags" in features:
                    self.tag_field = "ner_tags"
                    self.label_list = features["ner_tags"].feature.names
                elif features and "tags" in features:
                    self.tag_field = "tags"
                    self.label_list = features["tags"].feature.names
                else:
                    # In streaming mode, peek a sample to infer field name
                    if use_hf_streaming:
                        try:
                            _it = iter(self.ds)
                            _first = next(_it)
                            if "ner_tags" in _first:
                                self.tag_field = "ner_tags"
                            elif "tags" in _first:
                                self.tag_field = "tags"
                        except Exception:
                            pass
                    # Fallback canonical list
                    self.label_list = [
                        "O",
                        "B-corporation","I-corporation",
                        "B-creative-work","I-creative-work",
                        "B-group","I-group",
                        "B-location","I-location",
                        "B-person","I-person",
                        "B-product","I-product",
                    ]
                    if self.tag_field is None:
                        self.tag_field = "ner_tags"
            except Exception:
                self.label_list = [
                    "O",
                    "B-corporation","I-corporation",
                    "B-creative-work","I-creative-work",
                    "B-group","I-group",
                    "B-location","I-location",
                    "B-person","I-person",
                    "B-product","I-product",
                ]
                if self.tag_field is None:
                    self.tag_field = "ner_tags"

        self.label2id = {l:i for i,l in enumerate(self.label_list)}
        self.id2label = {i:l for l,i in self.label2id.items()}

        # Materialize streaming to enable __len__/__getitem__ with indices
        self._cached = None
        if use_hf_streaming:
            self._cached = list(self.ds)

    def __len__(self):
        if self.use_hf_streaming:
            return len(self._cached)
        return len(self.ds)

    def _get_item(self, idx: int) -> Dict:
        if self.use_hf_streaming:
            return self._cached[idx]
        else:
            return self.ds[idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._get_item(idx)
        tokens: List[str] = item["tokens"]
        ner_tags: List[int] = item[self.tag_field]

        ids: List[int] = []
        labels: List[int] = []

        if self.add_bos:
            ids.append(self.bos_token)
            labels.append(self.ignore_index)

        for tok, tag_id in zip(tokens, ner_tags):
            b = text_to_bytes(tok)
            if not b:
                continue
            reserve = 1 if self.add_eos else 0
            if len(ids) + len(b) + reserve > self.max_len:
                break
            ids.append(b[0]); labels.append(tag_id)
            for bb in b[1:]:
                ids.append(bb); labels.append(self.ignore_index)
            if len(ids) + 1 + reserve <= self.max_len:
                ids.append(ord(' ')); labels.append(self.ignore_index)

        if self.add_eos and len(ids) < self.max_len:
            ids.append(self.eos_token); labels.append(self.ignore_index)

        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids.extend([self.pad_token]*pad_len)
            labels.extend([self.ignore_index]*pad_len)
        else:
            ids = ids[:self.max_len]
            labels = labels[:self.max_len]

        attention_mask = [1 if t != self.pad_token else 0 for t in ids]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

def wnut_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
