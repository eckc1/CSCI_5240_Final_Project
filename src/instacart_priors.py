from collections import Counter, defaultdict
import itertools
import pandas as pd

class InstacartPriorBuilder:
    def __init__(self, instacart_bundle: dict | None):
        self.bundle = instacart_bundle
        self.product_name_lookup = {}
        self.cooccur_scores = {}
        self.global_product_frequency = Counter()

        if self.bundle is not None:
            products = self.bundle["products"]
            self.product_name_lookup = dict(zip(products["product_id"], products["product_name"]))

    def build(self, max_orders: int = 20000):
        if self.bundle is None:
            return

        prior = self.bundle["prior"][["order_id", "product_id"]].copy()
        sampled_orders = prior["order_id"].drop_duplicates().head(max_orders)
        prior = prior[prior["order_id"].isin(sampled_orders)]

        grouped = prior.groupby("order_id")["product_id"].apply(list)

        pair_counter = Counter()
        product_counter = Counter()

        for product_list in grouped:
            unique_products = sorted(set(product_list))
            for p in unique_products:
                product_counter[p] += 1
            for a, b in itertools.combinations(unique_products, 2):
                pair_counter[(a, b)] += 1

        self.global_product_frequency = product_counter

        cooccur_scores = {}
        for (a, b), pair_count in pair_counter.items():
            denom = max(1, min(product_counter[a], product_counter[b]))
            score = pair_count / denom
            cooccur_scores[(a, b)] = score
            cooccur_scores[(b, a)] = score

        self.cooccur_scores = cooccur_scores

    def map_foods_to_instacart(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
        if self.bundle is None or candidate_df.empty:
            out = candidate_df.copy()
            out["instacart_match_name"] = None
            out["instacart_product_id"] = None
            out["instacart_department"] = None
            return out

        products = self.bundle["products"].copy()
        products["product_name_l"] = products["product_name"].str.lower()

        out_rows = []
        for _, row in candidate_df.iterrows():
            query = str(row["item_name"]).lower()
            parts = [p for p in query.replace(",", " ").split() if len(p) > 2]

            subset = products
            if parts:
                mask = products["product_name_l"].apply(lambda x: sum(1 for p in parts if p in x) > 0)
                subset = products[mask].copy()

            if len(subset) == 0:
                row2 = row.copy()
                row2["instacart_match_name"] = None
                row2["instacart_product_id"] = None
                row2["instacart_department"] = None
                out_rows.append(row2)
                continue

            subset["match_score"] = subset["product_name_l"].apply(lambda x: sum(1 for p in parts if p in x))
            best = subset.sort_values(["match_score", "product_id"], ascending=[False, True]).iloc[0]

            row2 = row.copy()
            row2["instacart_match_name"] = best["product_name"]
            row2["instacart_product_id"] = int(best["product_id"])
            row2["instacart_department"] = best.get("department")
            out_rows.append(row2)

        return pd.DataFrame(out_rows)

    def add_basket_prior_scores(self, candidate_df: pd.DataFrame, seed_terms: list[str]) -> pd.DataFrame:
        out = candidate_df.copy()
        if self.bundle is None or out.empty:
            out["basket_prior_score"] = 0.0
            return out

        products = self.bundle["products"].copy()
        products["product_name_l"] = products["product_name"].str.lower()

        seed_ids = set()
        for term in seed_terms:
            term_l = term.lower()
            subset = products[products["product_name_l"].str.contains(term_l, na=False)]
            if len(subset):
                seed_ids.update(subset["product_id"].head(10).tolist())

        prior_scores = []
        for _, row in out.iterrows():
            pid = row.get("instacart_product_id")
            if pd.isna(pid):
                prior_scores.append(0.0)
                continue

            pid = int(pid)
            score = 0.0
            for sid in seed_ids:
                score += self.cooccur_scores.get((pid, sid), 0.0)
            prior_scores.append(score)

        out["basket_prior_score"] = prior_scores
        return out
