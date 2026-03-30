from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def evaluate_clusters(X: np.ndarray, labels_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows: List[dict] = []

    for name, labels in labels_dict.items():
        if len(np.unique(labels)) < 2:
            sil, ch = -1.0, -1.0
        else:
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)

        rows.append(
            {
                "model": name,
                "silhouette": float(sil),
                "calinski_harabasz": float(ch),
                "n_clusters": int(len(np.unique(labels))),
            }
        )

    return pd.DataFrame(rows).sort_values(["silhouette", "calinski_harabasz"], ascending=False)


def cluster_curve(X: np.ndarray, cluster_fn, k_values=range(2, 11)) -> pd.DataFrame:
    rows: List[dict] = []
    for k in k_values:
        labels = cluster_fn(X, k)
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        rows.append({"k": k, "silhouette": float(sil), "calinski_harabasz": float(ch)})
    return pd.DataFrame(rows)
