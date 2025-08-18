
from typing import List, Dict, Iterator, Optional
import re
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset

try:
    from datasets import VerificationMode
    _NO_CHECKS = VerificationMode.NO_CHECKS
except Exception:
    _NO_CHECKS = "no_checks"

from utils.rouge_l import greedy_oracle_indices, rouge1_f1, rouge_l_f1

def _sent_split(text: str) -> List[str]:
    # Lightweight sentence splitter; keeps punctuation, trims whitespace, drops empties.
    # You can swap this with nltk/pysbd if you prefer, but this avoids extra deps.
    parts = re.split(r'(?<=[\.!?])\s+', text.strip())
    sents = [s.strip() for s in parts if s and not s.isspace()]
    return sents

def _to_bytes(s: str) -> List[int]:
    return list(s.encode("utf-8", errors="ignore"))

def _greedy_with_metric(sentences: List[str], target: str, top_k: int, metric_fn, max_tokens: Optional[int] = None) -> List[int]:
    selected = []
    used = set()
    cur = ""
    def within_budget(text: str) -> bool:
        return True if max_tokens is None else (len(text.split()) <= max_tokens)
    for _ in range(min(top_k, len(sentences))):
        best_gain, best_idx, best_text = 0.0, -1, cur
        base_score = metric_fn(cur, target)
        for i, s in enumerate(sentences):
            if i in used: continue
            cand = (cur + " " + s).strip() if cur else s
            if not within_budget(cand): continue
            gain = metric_fn(cand, target) - base_score
            if gain > best_gain:
                best_gain, best_idx, best_text = gain, i, cand
        if best_idx == -1: break
        used.add(best_idx); selected.append(best_idx); cur = best_text
    return selected


class XSumExtractiveIterable(IterableDataset):
    """
    Streams XSum from HF Parquet branch, converts each article to a UTF-8 byte sequence,
    computes sentence spans, and labels sentences using greedy ROUGE-L against the gold summary.
    Yields tensors already padded to fixed shapes so default collate can stack batches.
    """
    def __init__(
        self,
        split: str,
        dataset_name: str = "EdinburghNLP/xsum",
        dataset_revision: str = "refs/convert/parquet",
        use_hf_streaming: bool = True,
        verification_no_checks: bool = True,
        src_field: str = "document",
        tgt_field: str = "summary",
        src_max_len: int = 2048,
        pad_token: int = 255,
        add_space_between_sentences: bool = True,
        max_sentences: int = 50,
        top_k: int = 3,
        max_summary_tokens: Optional[int] = None,
        oracle_metric: str = 'rouge1',
    ):
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_revision = dataset_revision
        self.use_hf_streaming = use_hf_streaming
        self.verification_no_checks = verification_no_checks
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.src_max_len = src_max_len
        self.pad_token = pad_token
        self.add_space_between_sentences = add_space_between_sentences
        self.max_sentences = max_sentences
        self.top_k = top_k
        self.max_summary_tokens = max_summary_tokens
        self.oracle_metric = oracle_metric

    def _load_iter(self):
        return load_dataset(
            self.dataset_name,
            split=self.split,
            revision=self.dataset_revision,
            streaming=self.use_hf_streaming,
            verification_mode=_NO_CHECKS if self.verification_no_checks else None,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ds = self._load_iter()

        for ex in ds:
            doc = ex[self.src_field]
            ref = ex[self.tgt_field]

            # Sentence split and cap
            sents = _sent_split(doc)[: self.max_sentences]

            # Derive oracle labels (indices of selected sentences)
            if len(sents) == 0:
                continue
            chosen = (_greedy_with_metric(sents, ref, top_k=self.top_k, metric_fn=rouge1_f1, max_tokens=self.max_summary_tokens)
                  if (getattr(self, 'oracle_metric', 'rougeL').lower() == 'rouge1')
                  else greedy_oracle_indices(sents, ref, top_k=self.top_k, max_tokens=self.max_summary_tokens))

            # Build byte sequence and sentence spans within it
            ids: List[int] = []
            spans: List[List[int]] = []  # [start, end) per sentence (capped by src_max_len)
            for si, sent in enumerate(sents):
                sb = _to_bytes(sent)
                if len(ids) + len(sb) + (1 if self.add_space_between_sentences else 0) > self.src_max_len:
                    break
                start = len(ids)
                ids.extend(sb)
                end = len(ids)
                spans.append([start, end])
                if self.add_space_between_sentences and end < self.src_max_len:
                    ids.append(ord(' '))
            # Re-cap labels/spans to those actually included
            sent_count = len(spans)
            if sent_count == 0:
                continue
            labels = [1 if i in set(chosen) and i < sent_count else 0 for i in range(sent_count)]

            # Pad byte sequence
            if len(ids) < self.src_max_len:
                ids.extend([self.pad_token] * (self.src_max_len - len(ids)))
            else:
                ids = ids[: self.src_max_len]

            # Pad spans to fixed count
            maxS = self.max_sentences
            span_pad = [[-1, -1]] * (maxS - sent_count)
            spans_padded = spans + span_pad
            sent_mask = [1]*sent_count + [0]*(maxS - sent_count)
            labels_padded = labels + [0]*(maxS - sent_count)

            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor([1 if t != self.pad_token else 0 for t in ids], dtype=torch.long),
                "spans": torch.tensor(spans_padded, dtype=torch.long),   # (max_sentences, 2)
                "sent_mask": torch.tensor(sent_mask, dtype=torch.long),  # (max_sentences,)
                "labels": torch.tensor(labels_padded, dtype=torch.float),# (max_sentences,) for BCE
                "gold_summary": ref,                                     # keep as plain string for metrics
            }
