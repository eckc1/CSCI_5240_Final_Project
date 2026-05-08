import pandas as pd

DEFAULT_PRICE_MAP = {
    "chicken": 9.5,
    "breast": 9.5,
    "egg": 4.0,
    "yogurt": 5.5,
    "oat": 4.0,
    "rice": 2.8,
    "spinach": 2.3,
    "broccoli": 2.8,
    "banana": 1.8,
    "berries": 4.7,
    "strawberry": 4.0,
    "blueberry": 5.0,
    "salmon": 11.0,
    "tofu": 2.5,
    "beans": 1.5,
    "bread": 3.2,
    "avocado": 1.7,
}

def estimate_price(item_name: str, category: str | None = None) -> float:
    name = str(item_name).lower()
    for k, v in DEFAULT_PRICE_MAP.items():
        if k in name:
            return float(v)

    cat = str(category).lower() if category is not None else ""
    if "vegetable" in cat or "produce" in cat:
        return 2.5
    if "dairy" in cat:
        return 4.5
    if "meat" in cat or "poultry" in cat:
        return 9.0
    return 3.5
