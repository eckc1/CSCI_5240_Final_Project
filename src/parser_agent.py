import re
from src.models import UserConstraints
from src.llm import chat_json

class ConstraintParserAgent:
    def parse(self, user_query: str, use_llm: bool = False) -> UserConstraints:
        if use_llm:
            out = self._parse_with_llm(user_query)
            if out is not None:
                try:
                    return UserConstraints(**out)
                except Exception:
                    pass
        return self._parse_heuristic(user_query)

    def _parse_with_llm(self, text: str):
        system = """
Extract grocery shopping constraints.
Return JSON with keys:
budget, min_protein_g, max_calories, meals_breakfast, meals_lunch, meals_dinner,
preferred_items, avoid_items, dietary_tags_required, low_added_sugar.
"""
        return chat_json(system, text)

    def _parse_heuristic(self, text: str) -> UserConstraints:
        t = text.lower()

        budget = None
        m = re.search(r"\$ ?(\d+(?:\.\d+)?)", t)
        if m:
            budget = float(m.group(1))

        min_protein = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*g\s+protein", t)
        if m:
            min_protein = float(m.group(1))

        def extract_meal_count(meal):
            for p in [rf"(\d+)\s+{meal}s?", rf"for\s+(\d+)\s+{meal}s?"]:
                mm = re.search(p, t)
                if mm:
                    return int(mm.group(1))
            return 0

        dietary = []
        for tag in ["vegetarian", "vegan", "gluten-free", "high-protein", "keto"]:
            if tag in t:
                dietary.append(tag)

        low_sugar = "low added sugar" in t or "low sugar" in t

        preferred_items = []
        for marker in ["prefer ", "include ", "with "]:
            if marker in t:
                chunk = t.split(marker, 1)[1]
                chunk = re.split(r"\. | avoid | under \$| keep ", chunk)[0]
                preferred_items = [x.strip() for x in re.split(r",| and ", chunk) if x.strip()]
                break

        avoid_items = []
        for marker in ["avoid ", "no "]:
            if marker in t:
                chunk = t.split(marker, 1)[1]
                chunk = re.split(r"\. | under \$| keep | prefer ", chunk)[0]
                avoid_items.extend([x.strip() for x in re.split(r",| and ", chunk) if x.strip()])

        return UserConstraints(
            budget=budget,
            min_protein_g=min_protein,
            meals_breakfast=extract_meal_count("breakfast"),
            meals_lunch=extract_meal_count("lunch"),
            meals_dinner=extract_meal_count("dinner"),
            preferred_items=list(dict.fromkeys(preferred_items)),
            avoid_items=list(dict.fromkeys(avoid_items)),
            dietary_tags_required=dietary,
            low_added_sugar=low_sugar,
        )
