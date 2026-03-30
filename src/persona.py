from __future__ import annotations

import pandas as pd


def build_persona_report(customer_features: pd.DataFrame, labels, cluster_name: str) -> pd.DataFrame:
    frame = customer_features.copy()
    frame["cluster"] = labels

    profile = (
        frame.groupby("cluster")
        .agg(
            users=("user_id", "count"),
            avg_orders=("total_orders", "mean"),
            avg_items=("avg_items_per_order", "mean"),
            reorder_ratio=("reordered_ratio", "mean"),
            avg_days_between=("avg_days_between", "mean"),
        )
        .reset_index()
    )

    profile["persona"] = profile.apply(
        lambda r: (
            "Loyal Bulk Buyers"
            if r["reorder_ratio"] > 0.5 and r["avg_items"] > 4
            else "Frequent Essentials Shoppers"
            if r["avg_orders"] > profile["avg_orders"].median()
            else "Opportunistic Mix Shoppers"
        ),
        axis=1,
    )
    profile["model"] = cluster_name
    return profile
