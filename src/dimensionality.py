from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:  # pragma: no cover
    UMAP_AVAILABLE = False


def compute_embeddings(X: pd.DataFrame, random_state: int = 42) -> Dict[str, np.ndarray]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Xs)
    pca3 = PCA(n_components=3, random_state=random_state).fit_transform(Xs)

    tsne2 = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto").fit_transform(Xs)

    embeddings: Dict[str, np.ndarray] = {
        "pca_2d": pca2,
        "pca_3d": pca3,
        "tsne_2d": tsne2,
    }

    if UMAP_AVAILABLE:
        embeddings["umap_2d"] = umap.UMAP(n_components=2, random_state=random_state).fit_transform(Xs)
        embeddings["umap_3d"] = umap.UMAP(n_components=3, random_state=random_state).fit_transform(Xs)

    return embeddings
