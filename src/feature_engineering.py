from __future__ import annotations

import pandas as pd


def build_customer_feature_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"]
    op = data["order_products"]

    merged = op.merge(orders, on="order_id", how="left")

    per_user = (
        merged.groupby("user_id")
        .agg(
            total_orders=("order_id", "nunique"),
            basket_items=("product_id", "count"),
            unique_products=("product_id", "nunique"),
            reordered_ratio=("reordered", "mean"),
            avg_cart_pos=("add_to_cart_order", "mean"),
            avg_order_hour=("order_hour_of_day", "mean"),
            avg_days_between=("days_since_prior_order", "mean"),
            order_dow_mode=("order_dow", lambda s: s.mode().iloc[0] if not s.mode().empty else 0),
        )
        .reset_index()
    )

    per_user["avg_items_per_order"] = per_user["basket_items"] / per_user["total_orders"].clip(lower=1)
    per_user = per_user.fillna(0)
    return per_user


def build_market_basket_matrix(data: dict[str, pd.DataFrame], max_products: int = 120) -> pd.DataFrame:
    op = data["order_products"]
    top_products = op["product_id"].value_counts().head(max_products).index
    slim = op[op["product_id"].isin(top_products)].copy()
    slim["present"] = 1
    basket = slim.pivot_table(
        index="order_id",
        columns="product_id",
        values="present",
        aggfunc="max",
        fill_value=0,
    )
    basket.columns = [f"product_{c}" for c in basket.columns]
    return basket.astype(int)
