from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from path_utils import ensure_dirs
from src.clustering import fit_cluster_models
from src.data_loader import load_instacart_tables
from src.dimensionality import compute_embeddings
from src.evaluation import cluster_curve, evaluate_clusters
from src.feature_engineering import build_customer_feature_matrix, build_market_basket_matrix
from src.persona import build_persona_report
from src.recommendation import mine_rules_apriori, mine_rules_fpgrowth


def run_training(n_clusters: int = 6) -> dict:
    paths = ensure_dirs()
    data = load_instacart_tables(paths["RAW"])

    customer_features = build_customer_feature_matrix(data)
    X = customer_features.drop(columns=["user_id"])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    embeddings = compute_embeddings(X)
    labels_dict = fit_cluster_models(Xs, n_clusters=n_clusters)
    eval_df = evaluate_clusters(Xs, labels_dict)

    best_cluster_model = eval_df.iloc[0]["model"]
    best_labels = labels_dict[best_cluster_model]

    basket = build_market_basket_matrix(data)
    apriori_rules = mine_rules_apriori(basket, min_support=0.01)
    fpgrowth_rules = mine_rules_fpgrowth(basket, min_support=0.01)

    persona_df = build_persona_report(customer_features, best_labels, cluster_name=best_cluster_model)

    # Validation-style curve for KMeans across K.
    curve_df = cluster_curve(
        Xs,
        cluster_fn=lambda Z, k: fit_cluster_models(Z, n_clusters=k)["kmeans"],
        k_values=range(2, 11),
    )

    models_dir = paths["MODELS"]
    charts_dir = paths["CHARTS"]

    # Persist artifacts.
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump({"labels": labels_dict, "best_model": best_cluster_model}, models_dir / "cluster_artifacts.pkl")

    eval_df.to_csv(models_dir / "cluster_evaluation.csv", index=False)
    curve_df.to_csv(models_dir / "kmeans_validation_curve.csv", index=False)
    persona_df.to_csv(models_dir / "business_persona_report.csv", index=False)

    if not apriori_rules.empty:
        apriori_rules.head(200).to_csv(models_dir / "apriori_rules_top200.csv", index=False)
    if not fpgrowth_rules.empty:
        fpgrowth_rules.head(200).to_csv(models_dir / "fpgrowth_rules_top200.csv", index=False)

    emb2 = embeddings["pca_2d"]
    plt.figure(figsize=(8, 6))
    plt.scatter(emb2[:, 0], emb2[:, 1], c=best_labels, s=8, cmap="tab10", alpha=0.8)
    plt.title(f"Project 12 - Customer Segments ({best_cluster_model})")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    plt.savefig(charts_dir / "customer_segments_pca2d.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(curve_df["k"], curve_df["silhouette"], marker="o", label="Silhouette")
    plt.plot(curve_df["k"], curve_df["calinski_harabasz"], marker="s", label="Calinski-Harabasz")
    plt.title("Project 12 - KMeans Selection Curves")
    plt.xlabel("Number of Clusters (K)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "kmeans_selection_curves.png", dpi=180)
    plt.close()

    payload = {
        "best_cluster_model": best_cluster_model,
        "cluster_evaluation": eval_df.to_dict(orient="records"),
        "persona_rows": len(persona_df),
        "apriori_rules": int(len(apriori_rules)),
        "fpgrowth_rules": int(len(fpgrowth_rules)),
    }

    with open(models_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    result = run_training(n_clusters=6)
    print(json.dumps(result, indent=2))
