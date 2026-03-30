from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_instacart_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    expected = {
        "orders": "orders.csv",
        "order_products": "order_products__prior.csv",
        "products": "products.csv",
        "departments": "departments.csv",
        "aisles": "aisles.csv",
    }

    data: dict[str, pd.DataFrame] = {}
    found_all = True
    for key, filename in expected.items():
        p = raw_dir / filename
        if p.exists():
            data[key] = pd.read_csv(p)
        else:
            found_all = False

    if found_all:
        return data

    # Synthetic fallback for quick experimentation and app demos.
    rng = np.random.default_rng(42)
    n_customers = 4500
    n_products = 300
    n_orders = 35000

    orders = pd.DataFrame(
        {
            "order_id": np.arange(1, n_orders + 1),
            "user_id": rng.integers(1, n_customers + 1, size=n_orders),
            "order_number": rng.integers(1, 35, size=n_orders),
            "order_dow": rng.integers(0, 7, size=n_orders),
            "order_hour_of_day": rng.integers(0, 24, size=n_orders),
            "days_since_prior_order": np.clip(rng.normal(8, 5, n_orders), 0, 30),
        }
    )

    basket_sizes = rng.integers(1, 9, size=n_orders)
    rows = []
    for oid, size in zip(orders["order_id"].values, basket_sizes):
        products = rng.choice(np.arange(1, n_products + 1), size=size, replace=False)
        for idx, pid in enumerate(products, start=1):
            rows.append((oid, int(pid), idx, int(rng.choice([0, 1], p=[0.82, 0.18]))))

    order_products = pd.DataFrame(
        rows,
        columns=["order_id", "product_id", "add_to_cart_order", "reordered"],
    )

    products = pd.DataFrame(
        {
            "product_id": np.arange(1, n_products + 1),
            "product_name": [f"product_{i}" for i in range(1, n_products + 1)],
            "aisle_id": rng.integers(1, 35, size=n_products),
            "department_id": rng.integers(1, 12, size=n_products),
        }
    )

    data = {
        "orders": orders,
        "order_products": order_products,
        "products": products,
        "departments": pd.DataFrame({"department_id": np.arange(1, 12)}),
        "aisles": pd.DataFrame({"aisle_id": np.arange(1, 35)}),
    }
    return data
