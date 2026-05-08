import pandas as pd
from src.models import UserConstraints, Plan
from src.llm import chat_json

class ExplainerAgent:
    def explain(self, user_query: str, constraints: UserConstraints, plan: Plan, basket_df: pd.DataFrame, use_llm: bool = False) -> str:
        if use_llm:
            out = chat_json(
                "Explain the resulting grocery basket. Return JSON with key explanation.",
                f"User query: {user_query}\nConstraints: {constraints.model_dump()}\nPlan: {plan.model_dump()}\nBasket: {basket_df.to_dict(orient='records')}"
            )
            if out and "explanation" in out:
                return out["explanation"]

        if basket_df.empty:
            return "No feasible basket was found."

        total = basket_df["price"].sum()
        protein = basket_df["protein_g"].fillna(0.0).sum()
        calories = basket_df["calories"].fillna(0.0).sum()
        items = ", ".join(basket_df["item_name"].head(6).tolist())

        return (
            f"This basket combines live USDA nutrition data with Instacart basket structure. "
            f"It selects {len(basket_df)} items totaling about ${total:.2f}, {protein:.1f} g protein, "
            f"and {calories:.0f} calories. Representative items include {items}. "
            f"The optimizer favored foods matching your preferred items, penalized added sugar when requested, "
            f"and boosted combinations that resemble realistic historical grocery baskets."
        )
