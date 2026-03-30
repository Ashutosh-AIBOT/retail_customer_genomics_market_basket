from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


def mine_rules_apriori(basket: pd.DataFrame, min_support: float = 0.01) -> pd.DataFrame:
    freq = apriori(basket.astype(bool), min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.2)
    if rules.empty:
        return rules
    return rules.sort_values(["lift", "confidence"], ascending=False)


def mine_rules_fpgrowth(basket: pd.DataFrame, min_support: float = 0.01) -> pd.DataFrame:
    freq = fpgrowth(basket.astype(bool), min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.2)
    if rules.empty:
        return rules
    return rules.sort_values(["lift", "confidence"], ascending=False)
