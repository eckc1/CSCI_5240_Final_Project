import requests
import pandas as pd

class USDAFoodDataClient:
    BASE_URL = "https://api.nal.usda.gov/fdc/v1"

    def __init__(self, api_key: str | None):
        self.api_key = api_key

    def search_foods(self, query: str, page_size: int = 25, data_types: list[str] | None = None) -> list[dict]:
        if not self.api_key:
            raise ValueError("USDA_API_KEY is not set.")

        url = f"{self.BASE_URL}/foods/search"
        payload = {
            "query": query,
            "pageSize": page_size,
        }
        if data_types:
            payload["dataType"] = data_types

        resp = requests.post(
            url,
            params={"api_key": self.api_key},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("foods", [])

    @staticmethod
    def foods_to_dataframe(foods: list[dict]) -> pd.DataFrame:
        rows = []
        for food in foods:
            nutrients = food.get("foodNutrients", []) or []
            nutrient_map = {}
            for n in nutrients:
                name = (n.get("nutrientName") or n.get("name") or "").lower()
                val = n.get("value")
                if name:
                    nutrient_map[name] = val

            def get_nutrient(possible_names, default=0.0):
                for name in possible_names:
                    for k, v in nutrient_map.items():
                        if name in k and v is not None:
                            return float(v)
                return default

            rows.append({
                "fdc_id": food.get("fdcId"),
                "item_name": food.get("description"),
                "brand_name": food.get("brandName"),
                "data_type": food.get("dataType"),
                "category": food.get("foodCategory"),
                "calories": get_nutrient(["energy"]),
                "protein_g": get_nutrient(["protein"]),
                "carbs_g": get_nutrient(["carbohydrate"]),
                "fat_g": get_nutrient(["total lipid", "fat"]),
                "fiber_g": get_nutrient(["fiber"]),
                "sugar_g": get_nutrient(["sugars"]),
                "added_sugar_g": get_nutrient(["added sugars"]),
                "serving_size": food.get("servingSize"),
                "serving_size_unit": food.get("servingSizeUnit"),
                "ingredients": food.get("ingredients"),
                "publication_date": food.get("publicationDate"),
            })
        return pd.DataFrame(rows)
