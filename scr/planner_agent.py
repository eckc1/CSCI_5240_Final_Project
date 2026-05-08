from src.models import UserConstraints, Plan
from src.llm import chat_json

class PlannerAgent:
    def create_plan(self, user_query: str, constraints: UserConstraints, use_llm: bool = False) -> Plan:
        if use_llm:
            system = "Create a short agent plan for a grocery optimization workflow. Return JSON with keys goal, steps, assumptions."
            out = chat_json(system, f"User query: {user_query}\nConstraints: {constraints.model_dump()}")
            if out:
                try:
                    return Plan(**out)
                except Exception:
                    pass

        return Plan(
            goal="Build a practical grocery basket using live nutrition data and historical basket priors.",
            steps=[
                "Parse constraints from the natural-language request.",
                "Query USDA FoodData Central for real foods related to the user goal.",
                "Score foods by preference match, protein value, sugar penalty, and cost estimate.",
                "Map foods to Instacart product categories when possible.",
                "Apply Instacart basket co-occurrence priors to favor realistic combinations.",
                "Optimize final basket under budget and nutrition constraints.",
                "Explain the result and tradeoffs."
            ],
            assumptions={
                "nutrition_source": "USDA FoodData Central",
                "behavioral_prior": "Instacart co-purchase structure",
                "pricing_source": "local default price mapping unless replaced"
            }
        )
