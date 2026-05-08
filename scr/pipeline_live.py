from src.parser_agent import ConstraintParserAgent
from src.planner_agent import PlannerAgent
from src.retriever_live import LiveGroceryRetriever
from src.optimizer import BasketOptimizer
from src.explainer_agent import ExplainerAgent
from src.instacart_priors import InstacartPriorBuilder


class GroceryOptimizationPipelineLive:
    def __init__(self, instacart_bundle, usda_api_key: str | None, max_prior_orders: int = 1000):
        self.parser = ConstraintParserAgent()
        self.planner = PlannerAgent()
        self.optimizer = BasketOptimizer()
        self.explainer = ExplainerAgent()

        self.prior_builder = None
        if instacart_bundle is not None:
            print("Initializing Instacart prior builder...")
            self.prior_builder = InstacartPriorBuilder(instacart_bundle)

            capped_orders = min(max_prior_orders, 5000)
            print(f"Building Instacart priors using {capped_orders} orders...")
            self.prior_builder.build(max_orders=capped_orders)
            print("Finished building Instacart priors.")

        self.retriever = LiveGroceryRetriever(
            usda_api_key=usda_api_key,
            prior_builder=self.prior_builder,
        )

    def run(self, user_query: str, use_llm: bool = False, top_k: int = 10):
        print("Parsing constraints...")
        constraints = self.parser.parse(user_query, use_llm=use_llm)

        print("Creating plan...")
        plan = self.planner.create_plan(user_query, constraints, use_llm=use_llm)

        print("Retrieving USDA foods...")
        candidates_df = self.retriever.retrieve(
            user_query,
            constraints,
            top_k=top_k,
        )

        print("Optimizing basket...")
        print(f"Candidate rows before optimize: {len(candidates_df)}")
        print(candidates_df[["item_name", "price", "protein_g", "calories", "utility_score"]].head(10))
        basket_df, details = self.optimizer.optimize(candidates_df, constraints)
        print("Finished optimization step.")

        print("Generating explanation...")
        explanation = self.explainer.explain(
            user_query,
            constraints,
            plan,
            basket_df,
            use_llm=use_llm,
        )

        summary = {
            "total_cost": float(basket_df["price"].sum()) if not basket_df.empty else 0.0,
            "num_items": int(len(basket_df)),
            "total_protein_g": float(basket_df["protein_g"].fillna(0.0).sum()) if not basket_df.empty else 0.0,
            "total_calories": float(basket_df["calories"].fillna(0.0).sum()) if not basket_df.empty else 0.0,
        }

        return {
            "constraints": constraints.model_dump(),
            "plan": plan.model_dump(),
            "candidates_df": candidates_df,
            "basket_df": basket_df,
            "optimization_details": details,
            "summary": summary,
            "explanation": explanation,
        }