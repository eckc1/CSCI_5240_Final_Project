from pathlib import Path
import pandas as pd

REQUIRED_FILES = [
    "products.csv",
    "orders.csv",
    "order_products__prior.csv",
    "aisles.csv",
    "departments.csv",
]

def instacart_data_available(instacart_dir: str) -> bool:
    p = Path(instacart_dir)
    return all((p / f).exists() for f in REQUIRED_FILES)

def load_instacart_data(instacart_dir: str) -> dict | None:
    p = Path(instacart_dir)
    if not instacart_data_available(instacart_dir):
        return None

    products = pd.read_csv(p / "products.csv")
    orders = pd.read_csv(p / "orders.csv")
    prior = pd.read_csv(p / "order_products__prior.csv")
    aisles = pd.read_csv(p / "aisles.csv")
    departments = pd.read_csv(p / "departments.csv")

    products = products.merge(aisles, on="aisle_id", how="left").merge(departments, on="department_id", how="left")
    return {
        "products": products,
        "orders": orders,
        "prior": prior,
        "aisles": aisles,
        "departments": departments,
    }
