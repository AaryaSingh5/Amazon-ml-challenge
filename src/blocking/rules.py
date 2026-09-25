"""Multi-Strategy Blocking Rules for Entity Resolution.

Implements complementary, high-recall blocking index structures:
1. Exact normalized name index
2. Suffix-stripped name index
3. Exact normalized address index
4. First-2-token prefix index
5. Rare token inverted index
6. Postal / PIN code + street number index
7. Character 3-gram signature index
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple, Optional, DefaultDict
from collections import defaultdict

# Common generic stopwords to exclude from inverted token index to avoid dense blocks
STOPWORDS = {
    "and", "the", "of", "in", "for", "on", "at", "to", "a", "an",
    "services", "service", "store", "shop", "center", "centre", "solutions",
    "enterprises", "group", "holdings", "trading", "international", "global",
    "corporation", "company", "limited", "pvt", "ltd", "inc", "llc",
}


def build_inverted_index(records: List[Dict[str, str]], key_fn) -> DefaultDict[str, List[int]]:
    """Builds an inverted index mapping blocking keys to record indices."""
    index: DefaultDict[str, List[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        keys = key_fn(rec)
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            if k and len(k) >= 2:
                index[k].append(idx)
    return index


class BlockingEngine:
    """Manages multi-strategy candidate generation and deduplication between S1 and S2/S3."""

    def __init__(self, max_candidates_per_entity: int = 100, max_block_size: int = 1500):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.max_block_size = max_block_size

        # Inverted indexes for target sources (S2 & S3)
        self.idx_exact_name: DefaultDict[str, List[int]] = defaultdict(list)
        self.idx_clean_name: DefaultDict[str, List[int]] = defaultdict(list)
        self.idx_exact_addr: DefaultDict[str, List[int]] = defaultdict(list)
        self.idx_prefix_tokens: DefaultDict[str, List[int]] = defaultdict(list)
        self.idx_rare_tokens: DefaultDict[str, List[int]] = defaultdict(list)
        self.idx_postal_digits: DefaultDict[str, List[int]] = defaultdict(list)
        self.idx_char_3grams: DefaultDict[str, List[int]] = defaultdict(list)

        self.target_records: List[Dict[str, str]] = []

    def fit_targets(self, s2_records: List[Dict[str, str]], s3_records: List[Dict[str, str]]) -> None:
        """Indexes all Source 2 and Source 3 candidate target records."""
        self.target_records = s2_records + s3_records
        logger_idx_count = len(self.target_records)

        for idx, rec in enumerate(self.target_records):
            # 1. Exact Name Key
            n_name = rec.get("normalized_name", "").strip()
            if n_name:
                self.idx_exact_name[n_name].append(idx)

            # 2. Suffix-Stripped Clean Name Key
            c_name = rec.get("clean_name_no_suffix", "").strip()
            if c_name:
                self.idx_clean_name[c_name].append(idx)

            # 3. Exact Address Key
            n_addr = rec.get("normalized_address", "").strip()
            if n_addr:
                self.idx_exact_addr[n_addr].append(idx)

            # 4. Name Prefix (First 2 tokens)
            tokens = c_name.split() if c_name else n_name.split()
            if len(tokens) >= 2:
                prefix = f"{tokens[0]}_{tokens[1]}"
                self.idx_prefix_tokens[prefix].append(idx)
            elif len(tokens) == 1:
                self.idx_prefix_tokens[tokens[0]].append(idx)

            # 5. Rare Name Tokens
            for tok in tokens:
                if len(tok) >= 3 and tok not in STOPWORDS:
                    self.idx_rare_tokens[tok].append(idx)

            # 6. Postal Code + Address Digits
            post = rec.get("postal_code", "").strip()
            digits = rec.get("address_digits", "").strip()
            if post and digits:
                first_digit = digits.split()[0]
                self.idx_postal_digits[f"{post}_{first_digit}"].append(idx)
            elif post:
                self.idx_postal_digits[post].append(idx)

            # 7. Character 3-Grams on Clean Name
            if c_name and len(c_name) >= 3:
                # Add first 3-gram and mid 3-gram for typo resilience
                g1 = c_name[:3]
                self.idx_char_3grams[g1].append(idx)
                if len(c_name) >= 6:
                    g2 = c_name[3:6]
                    self.idx_char_3grams[g2].append(idx)

    def generate_candidates_for_s1(self, s1_rec: Dict[str, str]) -> List[Tuple[str, Set[str]]]:
        """Generates candidate target entity IDs for a single Source 1 record.

        Returns:
            List of (target_entity_id, set_of_triggered_rules)
        """
        cand_rule_map: DefaultDict[int, Set[str]] = defaultdict(set)

        n_name = s1_rec.get("normalized_name", "").strip()
        c_name = s1_rec.get("clean_name_no_suffix", "").strip()
        n_addr = s1_rec.get("normalized_address", "").strip()
        post = s1_rec.get("postal_code", "").strip()
        digits = s1_rec.get("address_digits", "").strip()

        # Rule 1: Exact Normalized Name
        if n_name and n_name in self.idx_exact_name:
            targets = self.idx_exact_name[n_name]
            if len(targets) <= self.max_block_size:
                for t_idx in targets:
                    cand_rule_map[t_idx].add("rule_exact_name")

        # Rule 2: Suffix-Stripped Clean Name
        if c_name and c_name in self.idx_clean_name:
            targets = self.idx_clean_name[c_name]
            if len(targets) <= self.max_block_size:
                for t_idx in targets:
                    cand_rule_map[t_idx].add("rule_clean_name")

        # Rule 3: Exact Normalized Address
        if n_addr and n_addr in self.idx_exact_addr:
            targets = self.idx_exact_addr[n_addr]
            if len(targets) <= self.max_block_size:
                for t_idx in targets:
                    cand_rule_map[t_idx].add("rule_exact_addr")

        # Rule 4: Name Prefix Tokens
        tokens = c_name.split() if c_name else n_name.split()
        if len(tokens) >= 2:
            prefix = f"{tokens[0]}_{tokens[1]}"
            if prefix in self.idx_prefix_tokens:
                targets = self.idx_prefix_tokens[prefix]
                if len(targets) <= self.max_block_size:
                    for t_idx in targets:
                        cand_rule_map[t_idx].add("rule_prefix_tokens")
        elif len(tokens) == 1:
            tok = tokens[0]
            if tok in self.idx_prefix_tokens:
                targets = self.idx_prefix_tokens[tok]
                if len(targets) <= self.max_block_size:
                    for t_idx in targets:
                        cand_rule_map[t_idx].add("rule_prefix_tokens")

        # Rule 5: Rare Token Inverted Index
        for tok in tokens:
            if len(tok) >= 3 and tok not in STOPWORDS and tok in self.idx_rare_tokens:
                targets = self.idx_rare_tokens[tok]
                if len(targets) <= self.max_block_size:
                    for t_idx in targets:
                        cand_rule_map[t_idx].add("rule_rare_token")

        # Rule 6: Postal Code + Street Digits
        if post and digits:
            first_digit = digits.split()[0]
            key = f"{post}_{first_digit}"
            if key in self.idx_postal_digits:
                targets = self.idx_postal_digits[key]
                if len(targets) <= self.max_block_size:
                    for t_idx in targets:
                        cand_rule_map[t_idx].add("rule_postal_digits")

        # Rule 7: Character 3-Gram Overlap
        if c_name and len(c_name) >= 3:
            g1 = c_name[:3]
            if g1 in self.idx_char_3grams:
                targets = self.idx_char_3grams[g1]
                if len(targets) <= self.max_block_size:
                    for t_idx in targets:
                        cand_rule_map[t_idx].add("rule_char_3gram")

        # Format candidates and enforce per-entity cap
        results: List[Tuple[str, Set[str]]] = []
        
        # Sort candidates by number of triggered rules descending (strongest evidence first)
        sorted_cands = sorted(cand_rule_map.items(), key=lambda item: len(item[1]), reverse=True)
        for t_idx, triggered_rules in sorted_cands[:self.max_candidates_per_entity]:
            target_id = self.target_records[t_idx]["entity_id"]
            results.append((target_id, triggered_rules))

        return results
