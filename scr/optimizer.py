import itertools
import pandas as pd
from .models import UserConstraints


class BasketOptimizer:
    def optimize(self, candidates_df: pd.DataFrame, constraints: UserConstraints):
        df = candidates_df.copy().reset_index(drop=True)

        if df.empty:
            return df, {"status": "no_candidates"}

        required_numeric = ["price", "protein_g", "calories", "utility_score"]
        for col in required_numeric:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["price", "protein_g", "calories", "utility_score"]).copy()
        df = df[
            (df["price"] >= 0) &
            (df["protein_g"] >= 0) &
            (df["calories"] >= 0)
        ].copy()

        df = df.sort_values("utility_score", ascending=False).head(20).reset_index(drop=True)

        if df.empty:
            return df, {"status": "no_valid_candidates"}

        print(f"Optimizer received {len(df)} valid candidates.")

        # For grocery baskets, do not force one item per meal.
        # Just require a reasonable basket size.
        min_items = 3
        max_items = min(10, len(df))

        best_score = None
        best_indices = None
        checked = 0

        prices = df["price"].tolist()
        protein = df["protein_g"].tolist()
        calories = df["calories"].tolist()
        utility = df["utility_score"].tolist()

        all_indices = list(range(len(df)))

        print("Running brute-force subset search...")
        for r in range(min_items, max_items + 1):
            for combo in itertools.combinations(all_indices, r):
                checked += 1

                total_price = sum(prices[i] for i in combo)
                total_protein = sum(protein[i] for i in combo)
                total_calories = sum(calories[i] for i in combo)
                total_utility = sum(utility[i] for i in combo)

                if constraints.budget is not None and total_price > constraints.budget:
                    continue

                if constraints.min_protein_g is not None and total_protein < constraints.min_protein_g:
                    continue

                if constraints.max_calories is not None and total_calories > constraints.max_calories:
                    continue

                if best_score is None or total_utility > best_score:
                    best_score = total_utility
                    best_indices = combo

        print(f"Checked {checked} basket combinations.")

        if best_indices is None:
            # fallback: best affordable greedy subset
            fallback = df.copy()
            if constraints.budget is not None:
                fallback = fallback.sort_values("utility_score", ascending=False).copy()
                selected = []
                running_cost = 0.0
                for idx, row in fallback.iterrows():
                    if running_cost + row["price"] <= constraints.budget:
                        selected.append(idx)
                        running_cost += row["price"]
                basket = fallback.loc[selected].reset_index(drop=True)
            else:
                basket = df.head(min(5, len(df))).copy().reset_index(drop=True)

            details = {
                "status": "fallback_no_feasible_subset",
                "selected_count": int(len(basket)),
                "objective_value": None,
            }
            return basket, details

        basket = df.iloc[list(best_indices)].copy().reset_index(drop=True)

        details = {
            "status": "optimal_bruteforce",
            "selected_count": int(len(basket)),
            "objective_value": float(best_score),
        }

        return basket, details