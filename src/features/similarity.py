"""String similarity and pairwise comparison functions for entity resolution.

Implements edit distances, token set/sort ratios, Jaccard similarities,
n-gram overlaps, numeric token matching, and country comparisons.
"""

from __future__ import annotations

import re
import difflib
from typing import Set, List, Dict, Any, Optional

try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Computes normalized edit similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(s1, s2) / 100.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def token_sort_ratio(s1: str, s2: str) -> float:
    """Computes token sort similarity in [0.0, 1.0] (word-order invariant)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(s1, s2) / 100.0
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return difflib.SequenceMatcher(None, t1, t2).ratio()


def token_set_ratio(s1: str, s2: str) -> float:
    """Computes token set similarity in [0.0, 1.0] (handles subsets and acronyms)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return fuzz.token_set_ratio(s1, s2) / 100.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    intersection = set1.intersection(set2)
    if not intersection:
        return token_sort_ratio(s1, s2)
    inter_str = " ".join(sorted(intersection))
    diff1 = " ".join(sorted(set1 - intersection))
    diff2 = " ".join(sorted(set2 - intersection))
    comb1 = (inter_str + " " + diff1).strip()
    comb2 = (inter_str + " " + diff2).strip()
    r1 = difflib.SequenceMatcher(None, inter_str, comb1).ratio()
    r2 = difflib.SequenceMatcher(None, inter_str, comb2).ratio()
    r3 = difflib.SequenceMatcher(None, comb1, comb2).ratio()
    return max(r1, r2, r3)


def jaccard_similarity(s1: str, s2: str) -> float:
    """Computes word token Jaccard similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 and not tokens2:
        return 1.0
    union = tokens1.union(tokens2)
    if not union:
        return 0.0
    return len(tokens1.intersection(tokens2)) / len(union)


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Computes character n-gram Jaccard similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    grams1 = {s1[i:i+n] for i in range(max(0, len(s1) - n + 1))} if len(s1) >= n else {s1}
    grams2 = {s2[i:i+n] for i in range(max(0, len(s2) - n + 1))} if len(s2) >= n else {s2}
    union = grams1.union(grams2)
    if not union:
        return 0.0
    return len(grams1.intersection(grams2)) / len(union)


def numeric_tokens_jaccard(s1: str, s2: str) -> float:
    """Computes Jaccard overlap of numeric sequences in text (e.g. house numbers)."""
    digits1 = set(re.findall(r'\b\d+\b', s1))
    digits2 = set(re.findall(r'\b\d+\b', s2))
    if not digits1 and not digits2:
        return 0.5  # Neutral indicator when neither has digits
    if not digits1 or not digits2:
        return 0.0
    return len(digits1.intersection(digits2)) / len(digits1.union(digits2))


def postal_code_agreement(post1: str, post2: str) -> float:
    """Scores postal code agreement: 1.0 (match), 0.0 (mismatch), 0.5 (neutral/missing)."""
    p1 = str(post1).strip() if post1 else ""
    p2 = str(post2).strip() if post2 else ""
    if not p1 or not p2:
        return 0.5
    return 1.0 if p1 == p2 else 0.0


def length_features(s1: str, s2: str) -> Tuple[int, float, int]:
    """Computes absolute length difference, length ratio, and token count difference."""
    l1 = len(s1) if s1 else 0
    l2 = len(s2) if s2 else 0
    diff_abs = abs(l1 - l2)
    ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 1.0
    tc1 = len(s1.split()) if s1 else 0
    tc2 = len(s2.split()) if s2 else 0
    tc_diff = abs(tc1 - tc2)
    return diff_abs, float(round(ratio, 4)), tc_diff
