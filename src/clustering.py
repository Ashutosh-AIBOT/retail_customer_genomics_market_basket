from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture


def fit_cluster_models(X: np.ndarray, n_clusters: int = 6, random_state: int = 42) -> Dict[str, np.ndarray]:
    labels: Dict[str, np.ndarray] = {}

    labels["kmeans"] = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20).fit_predict(X)
    labels["agglomerative"] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
    labels["gmm"] = GaussianMixture(n_components=n_clusters, random_state=random_state).fit_predict(X)
    labels["spectral"] = SpectralClustering(
        n_clusters=n_clusters,
        random_state=random_state,
        assign_labels="kmeans",
        affinity="nearest_neighbors",
    ).fit_predict(X)

    return labels
