# Methodology & Modeling Strategy

## 1. Problem Definition
The objective is to map entities from a deduplicated reference source (**Source 1**) to their corresponding records in **Source 2** and **Source 3**. Each entity consists of:
- `entity_id`
- `business_name`
- `business_address`
- `country` (open-set: training contains US and India; test set may include France and other countries).

## 2. Pipeline Design Principles
1. **Precision Prioritization ($F_{0.5}$)**: Because $F_{0.5}$ weights precision higher than recall, candidate pairs must pass strict probability thresholds, especially for singletons.
2. **Leak-Free Validation**: Splitting is strictly performed at the Source 1 entity level, ensuring no entity overlaps between train and validation folds.
3. **Multi-Strategy Blocking**: To avoid missed matches, blocking combines multiple orthogonal rules (normalized exact match, token Jaccard overlap, character n-gram MinHash, country-partitioned fuzzy search) achieving $>98\%$ candidate recall while maintaining $O(N)$ candidate volume.
4. **No Prohibited Augmentation**: Strictly adheres to competition rules — zero external APIs, geocoders, web searches, or registry databases.
5. **Open-Set Country Support**: Country features are modeled symmetrically without hard-coding specific country lists.
