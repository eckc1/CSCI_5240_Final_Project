import re
import pandas as pd
from .usda_api import USDAFoodDataClient
from .price_estimator import estimate_price
from .models import UserConstraints


class LiveGroceryRetriever:
    def __init__(self, usda_api_key: str | None, prior_builder=None):
        self.usda = USDAFoodDataClient(usda_api_key)
        self.prior_builder = prior_builder

    def retrieve(
        self,
        query: str,
        constraints: UserConstraints,
        top_k: int = 25,
        use_instacart_priors: bool = True,
    ) -> pd.DataFrame:
        """
        Retrieve candidate foods from USDA, clean/filter them, optionally add
        Instacart prior scores, and return a ranked candidate table.
        """

        # Build better USDA query strings
        search_queries = self._build_search_queries(query, constraints)

        frames = []
        per_query_page_size = max(10, min(top_k, 25))

        for q in search_queries:
            try:
                foods = self.usda.search_foods(
                    query=q,
                    page_size=per_query_page_size,
                    data_types=["Foundation", "SR Legacy", "Survey (FNDDS)", "Branded"],
                )
                df_part = self.usda.foods_to_dataframe(foods)
                if not df_part.empty:
                    df_part["source_query"] = q
                    frames.append(df_part)
            except Exception as e:
                print(f"USDA query failed for '{q}': {e}")

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["fdc_id"], keep="first")

        if df.empty:
            return df

        # Standardize text fields
        for col in ["item_name", "brand_name", "category", "ingredients"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        df["search_text"] = (
            df["item_name"].str.lower() + " " +
            df["brand_name"].str.lower() + " " +
            df["category"].str.lower() + " " +
            df["ingredients"].str.lower()
        ).str.strip()

        # Numeric coercion
        numeric_cols = [
            "calories", "protein_g", "carbs_g", "fat_g",
            "fiber_g", "sugar_g", "added_sugar_g"
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Primary filters
        df = self._apply_avoid_filter(df, constraints)
        df = self._apply_junk_filter(df)
        df = self._apply_basic_nutrition_sanity_filter(df)

        if df.empty:
            return df

        # Add price estimate
        df["estimated_price"] = df.apply(
            lambda r: estimate_price(r["item_name"], r.get("category")),
            axis=1
        )

        # Preference / relevance features
        preferred_terms = [x.lower().strip() for x in constraints.preferred_items if x.strip()]
        df["preference_hits"] = df["search_text"].apply(
            lambda txt: sum(1 for term in preferred_terms if term in txt)
        )

        df["query_overlap_score"] = df["search_text"].apply(
            lambda txt: self._token_overlap_score(txt, preferred_terms)
        )

        # Penalize sugar if requested
        if constraints.low_added_sugar:
            df["sugar_penalty"] = df["added_sugar_g"].fillna(0.0) + 0.5 * df["sugar_g"].fillna(0.0)
        else:
            df["sugar_penalty"] = 0.0

        # Penalize branded / highly processed style results a bit
        df["processed_penalty"] = df["search_text"].apply(self._processed_food_penalty)

        # Reward fiber slightly, protein strongly, cheapness moderately
        # Penalize calories lightly to avoid very calorie-dense weird items dominating
        df["base_score"] = (
            0.12 * df["protein_g"]
            + 0.03 * df["fiber_g"]
            - 0.02 * df["estimated_price"]
            - 0.01 * df["calories"]
            - 0.04 * df["sugar_penalty"]
            - 0.20 * df["processed_penalty"]
            + 1.40 * df["preference_hits"]
            + 1.00 * df["query_overlap_score"]
        )

        # Duplicate / near-duplicate cleanup before Instacart enrichment
        df["canonical_name"] = df["item_name"].apply(self._canonicalize_name)
        df = self._deduplicate_candidates(df)

        # Optional Instacart priors
        if use_instacart_priors and self.prior_builder is not None and not df.empty:
            try:
                df = self.prior_builder.map_foods_to_instacart(df)
                df = self.prior_builder.add_basket_prior_scores(df, constraints.preferred_items)
            except Exception as e:
                print(f"Instacart prior step failed: {e}")
                df["instacart_match_name"] = None
                df["instacart_product_id"] = None
                df["instacart_department"] = None
                df["basket_prior_score"] = 0.0
        else:
            df["instacart_match_name"] = None
            df["instacart_product_id"] = None
            df["instacart_department"] = None
            df["basket_prior_score"] = 0.0

        # Final score
        df["utility_score"] = df["base_score"] + 1.25 * df["basket_prior_score"].fillna(0.0)

        # Encourage some category diversity by downweighting too many similar items
        df = self._soft_diversity_penalty(df)

        # Final sort and trim
        df["price"] = df["estimated_price"]
        df = df.sort_values("utility_score", ascending=False).head(top_k).reset_index(drop=True)

        return df

    def _build_search_queries(self, query: str, constraints: UserConstraints) -> list[str]:
        queries = []
        preferred = [x.strip() for x in constraints.preferred_items if x.strip()]

        if preferred:
            combined = " ".join(preferred[:8])
            queries.append(combined)

            # small grouped queries help recover better staples
            for item in preferred[:6]:
                queries.append(item)

        # fallback to original user query last
        queries.append(query)

        # de-duplicate while keeping order
        seen = set()
        out = []
        for q in queries:
            qq = q.strip()
            if qq and qq.lower() not in seen:
                seen.add(qq.lower())
                out.append(qq)

        return out[:8]

    def _apply_avoid_filter(self, df: pd.DataFrame, constraints: UserConstraints) -> pd.DataFrame:
        if not constraints.avoid_items:
            return df

        avoid_terms = [x.lower().strip() for x in constraints.avoid_items if x.strip()]
        if not avoid_terms:
            return df

        mask = ~df["search_text"].apply(lambda txt: any(term in txt for term in avoid_terms))
        return df[mask].copy()

    def _apply_junk_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        bad_terms = [
            "babyfood", "baby food", "infant", "formula", "supplement", "protein bar",
            "nutrition bar", "snack bar", "candy", "syrup", "dessert topping",
            "frosting", "oil,", "shortening", "lard", "alcohol", "beer", "wine",
            "liquor", "cocktail", "powdered drink", "energy drink"
        ]

        # Keep plain yogurt but allow branded yogurts only if not obviously dessert-like
        dessert_terms = ["cookie", "candy", "birthday cake", "cheesecake", "caramel"]

        mask = ~df["search_text"].apply(
            lambda txt: any(term in txt for term in bad_terms) or any(term in txt for term in dessert_terms)
        )

        return df[mask].copy()

    def _apply_basic_nutrition_sanity_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove obviously unhelpful candidates for grocery basket optimization.
        """
        df = df.copy()

        # Remove extreme calorie outliers or near-empty oddities
        df = df[(df["calories"] >= 0) & (df["calories"] <= 900)].copy()

        # Remove items that are nutritionally near-empty and not explicitly preferred
        df = df[
            ~(
                (df["protein_g"] < 1.0) &
                (df["fiber_g"] < 1.0) &
                (df["calories"] > 250)
            )
        ].copy()

        return df

    def _processed_food_penalty(self, txt: str) -> float:
        processed_terms = [
            "branded", "ready-to-eat", "flavored", "sweetened", "bar", "snack",
            "dessert", "frozen dinner", "instant"
        ]
        return float(sum(1 for term in processed_terms if term in txt))

    def _canonicalize_name(self, name: str) -> str:
        txt = str(name).lower()

        txt = re.sub(r"[^a-z0-9\s]", " ", txt)
        txt = re.sub(r"\b(lowfat|low fat|nonfat|plain|cooked|dry|enriched|junior|strained|mixed|blend|fat)\b", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()

        # Hand-tuned simplifications for common duplicates
        if "greek" in txt and "yogurt" in txt:
            return "greek yogurt"
        if "chicken" in txt:
            return "chicken"
        if "egg" in txt and "noodles" not in txt:
            return "eggs"
        if "oat" in txt:
            return "oats"
        if "rice" in txt:
            return "rice"
        if "spinach" in txt:
            return "spinach"
        if "broccoli" in txt:
            return "broccoli"
        if "banana" in txt:
            return "banana"
        if "berry" in txt or "berries" in txt:
            return "berries"

        return txt

    def _deduplicate_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only the best-scoring candidate per canonical name.
        """
        if df.empty:
            return df

        # Temporary ranking before final utility score exists
        temp_score = (
            0.10 * df["protein_g"]
            + 0.02 * df["fiber_g"]
            - 0.02 * df["estimated_price"]
            - 0.01 * df["calories"]
            - 0.04 * df["sugar_penalty"]
            + 1.20 * df["preference_hits"]
            + 0.80 * df["query_overlap_score"]
        )
        df = df.copy()
        df["_dedup_score"] = temp_score

        df = (
            df.sort_values("_dedup_score", ascending=False)
              .drop_duplicates(subset=["canonical_name"], keep="first")
              .drop(columns=["_dedup_score"])
              .reset_index(drop=True)
        )
        return df

    def _soft_diversity_penalty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downweight nearly identical categories so the optimizer does not get overwhelmed by one class
        """
        if df.empty:
            return df

        df = df.copy()
        group_counts = df["canonical_name"].map(df["canonical_name"].value_counts()).fillna(1)

        # Since dedup already collapses many items, this penalty is mild
        df["utility_score"] = df["utility_score"] - 0.05 * (group_counts - 1)

        # Also mildly penalize too many same-category items
        if "category" in df.columns:
            cat_counts = df["category"].fillna("").map(df["category"].fillna("").value_counts()).fillna(1)
            df["utility_score"] = df["utility_score"] - 0.01 * (cat_counts - 1)

        return df

    def _token_overlap_score(self, txt: str, preferred_terms: list[str]) -> float:
        if not preferred_terms:
            return 0.0

        txt_tokens = set(re.findall(r"[a-z0-9]+", txt.lower()))
        overlap = 0
        for term in preferred_terms:
            term_tokens = set(re.findall(r"[a-z0-9]+", term.lower()))
            if term_tokens and len(txt_tokens.intersection(term_tokens)) > 0:
                overlap += 1
        return float(overlap)