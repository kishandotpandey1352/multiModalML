
# Lightweight ROUGE utilities (no external deps).
# - rouge_l_f1: standard LCS-based ROUGE-L F1 (O(n*m))
# - rouge1_f1:  unigram F1 (fast proxy)
# - greedy_oracle_indices: greedy sentence selection using ROUGE-L

from typing import List

def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

def rouge1_f1(pred: str, ref: str) -> float:
    A = pred.split()
    B = ref.split()
    if not A or not B: 
        return 0.0
    from collections import Counter
    a, b = Counter(A), Counter(B)
    overlap = sum((a & b).values())
    prec = overlap / len(A)
    rec  = overlap / len(B)
    return _f1(prec, rec)

def _lcs_len(a: List[str], b: List[str]) -> int:
    # standard DP LCS
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = a[i-1]
        dpi, dpim1 = dp[i], dp[i-1]
        for j in range(1, m+1):
            if ai == b[j-1]:
                dpi[j] = dpim1[j-1] + 1
            else:
                dpi[j] = dpi[j-1] if dpi[j-1] >= dpim1[j] else dpim1[j]
    return dp[n][m]

def rouge_l_f1(pred: str, ref: str) -> float:
    A = pred.split()
    B = ref.split()
    if not A or not B:
        return 0.0
    lcs = _lcs_len(A, B)
    prec = lcs / len(A)
    rec  = lcs / len(B)
    return _f1(prec, rec)

def greedy_oracle_indices(sentences: List[str], target: str, top_k: int = 3, max_tokens: int | None = None) -> List[int]:
    """
    Greedy selection that maximizes ROUGE-L F1 against target.
    Optionally respects a token budget (by whitespace tokens).
    """
    selected: List[int] = []
    used   = set()
    cur    = ""

    def within_budget(text: str) -> bool:
        return True if max_tokens is None else (len(text.split()) <= max_tokens)

    for _ in range(min(top_k, len(sentences))):
        best_gain, best_idx, best_text = 0.0, -1, cur
        for i, s in enumerate(sentences):
            if i in used: 
                continue
            cand = (cur + " " + s).strip() if cur else s
            if not within_budget(cand):
                continue
            score_cand = rouge_l_f1(cand, target)
            gain = score_cand - rouge_l_f1(cur, target)
            if gain > best_gain:
                best_gain, best_idx, best_text = gain, i, cand
        if best_idx == -1:
            break
        used.add(best_idx)
        selected.append(best_idx)
        cur = best_text
    return selected
