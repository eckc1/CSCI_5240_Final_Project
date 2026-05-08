import time
import json
from itertools import combinations
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import settings
from src.instacart_loader import load_instacart_data, instacart_data_available
from src.pipeline_live import GroceryOptimizationPipelineLive


st.set_page_config(page_title="Live Grocery Optimizer", layout="wide")

st.title("Grocery Store Optimizer — USDA + Instacart")
st.caption("Agentic grocery optimization using live USDA nutrition search and Instacart basket priors.")


# Output directories
TEST_DIR = Path("testing")
PLOTS_DIR = TEST_DIR / "plots"
TABLES_DIR = TEST_DIR / "tables"
SINGLE_RUN_DIR = TEST_DIR / "single_runs"
FAILURE_DIR = TEST_DIR / "failure_cases"
for d in [TEST_DIR, PLOTS_DIR, TABLES_DIR, SINGLE_RUN_DIR, FAILURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Cached data loading only
@st.cache_data(show_spinner=True)
def cached_load_instacart(instacart_dir: str):
    return load_instacart_data(instacart_dir)



# Helper functions
def safe_yes_no(value):
    if value is None:
        return "NA"
    return "Yes" if bool(value) else "No"


def safe_num(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_text(x):
    if x is None:
        return ""
    return str(x).strip().lower()


def get_item_name_col(df: pd.DataFrame):
    return safe_col(df, ["item_name", "description", "food_description", "name"])


def get_price_col(df: pd.DataFrame):
    return safe_col(df, ["estimated_price", "price", "price_estimate"])


def get_protein_col(df: pd.DataFrame):
    return safe_col(df, ["protein_g", "protein"])


def get_calorie_col(df: pd.DataFrame):
    return safe_col(df, ["calories", "energy_kcal", "kcal"])


def get_sugar_col(df: pd.DataFrame):
    return safe_col(df, ["added_sugars_g", "sugar_g", "sugars_g", "added_sugar_g"])


def get_prior_col(df: pd.DataFrame):
    return safe_col(df, ["basket_prior_score", "prior_score"])


def get_overlap_col(df: pd.DataFrame):
    return safe_col(df, ["query_overlap_score", "overlap_score"])


def get_prefhit_col(df: pd.DataFrame):
    return safe_col(df, ["preference_hits", "preference_hit_count"])


def get_processed_col(df: pd.DataFrame):
    return safe_col(df, ["processed_penalty", "processing_penalty"])


def get_score_col(df: pd.DataFrame):
    return safe_col(df, ["score", "final_score", "utility_score"])


def extract_basket_text(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    text_parts = []
    for col in ["item_name", "description", "food_description", "ingredients", "search_text", "name"]:
        if col in df.columns:
            text_parts.extend(df[col].fillna("").astype(str).tolist())
    return " ".join(text_parts).lower()


def compute_constraint_metrics(result: dict) -> dict:
    constraints = result.get("constraints", {})
    basket_df = result.get("basket_df", pd.DataFrame()).copy()

    metrics = {}

    # Summary values
    total_cost = safe_num(result.get("summary", {}).get("total_cost", 0.0))
    total_protein = safe_num(result.get("summary", {}).get("total_protein_g", 0.0))
    total_calories = safe_num(result.get("summary", {}).get("total_calories", 0.0))

    # Budget
    budget = constraints.get("budget")
    metrics["budget_value"] = budget
    metrics["budget_met"] = (total_cost <= budget) if budget is not None else None
    metrics["budget_margin"] = (budget - total_cost) if budget is not None else None

    # Min protein
    min_protein = constraints.get("min_protein_g")
    metrics["min_protein_g"] = min_protein
    metrics["protein_target_met"] = (total_protein >= min_protein) if min_protein is not None else None
    metrics["protein_margin_g"] = (total_protein - min_protein) if min_protein is not None else None

    # Max calories
    max_calories = constraints.get("max_calories")
    metrics["max_calories"] = max_calories
    metrics["calorie_target_met"] = (total_calories <= max_calories) if max_calories is not None else None
    metrics["calorie_margin"] = (max_calories - total_calories) if max_calories is not None else None

    # Exclusions
    avoid_items = constraints.get("avoid_items", []) or []
    basket_text = extract_basket_text(basket_df)

    if avoid_items:
        violations = [term for term in avoid_items if normalize_text(term) in basket_text]
        metrics["exclusions_met"] = len(violations) == 0
        metrics["avoid_violation_terms"] = ", ".join(map(str, violations)) if violations else ""
        metrics["avoid_violation_count"] = len(violations)
    else:
        metrics["exclusions_met"] = None
        metrics["avoid_violation_terms"] = ""
        metrics["avoid_violation_count"] = None

    # Preference coverage
    preferred = constraints.get("preferred_items", []) or constraints.get("prefer_items", []) or []
    if preferred:
        hits = [term for term in preferred if normalize_text(term) in basket_text]
        metrics["preferred_hit_count"] = len(hits)
        metrics["preferred_hit_rate"] = len(hits) / max(len(preferred), 1)
    else:
        metrics["preferred_hit_count"] = None
        metrics["preferred_hit_rate"] = None

    checks = [
        metrics["budget_met"],
        metrics["protein_target_met"],
        metrics["calorie_target_met"],
        metrics["exclusions_met"],
    ]
    checks = [x for x in checks if x is not None]
    metrics["overall_constraint_satisfaction"] = all(checks) if checks else None

    return metrics


def compute_plausibility_metrics(basket_df: pd.DataFrame) -> dict:
    if basket_df is None or basket_df.empty:
        return {
            "plausibility_score": None,
            "avg_basket_prior_score": None,
            "avg_query_overlap_score": None,
            "avg_preference_hits": None,
            "avg_processed_penalty": None,
            "pairwise_name_diversity": None,
        }

    prior_col = get_prior_col(basket_df)
    overlap_col = get_overlap_col(basket_df)
    pref_col = get_prefhit_col(basket_df)
    processed_col = get_processed_col(basket_df)
    item_col = get_item_name_col(basket_df)

    avg_prior = basket_df[prior_col].mean() if prior_col else None
    avg_overlap = basket_df[overlap_col].mean() if overlap_col else None
    avg_pref = basket_df[pref_col].mean() if pref_col else None
    avg_processed = basket_df[processed_col].mean() if processed_col else None

    diversity = None
    if item_col:
        names = basket_df[item_col].fillna("").astype(str).tolist()
        unique_tokens = set()
        all_tokens = []
        for n in names:
            toks = [t for t in normalize_text(n).split() if len(t) > 2]
            all_tokens.extend(toks)
            unique_tokens.update(toks)
        diversity = len(unique_tokens) / max(len(all_tokens), 1) if all_tokens else None

    components = []
    for val in [avg_prior, avg_overlap, avg_pref]:
        if val is not None and pd.notna(val):
            components.append(float(val))
    if avg_processed is not None and pd.notna(avg_processed):
        components.append(float(-avg_processed))

    plausibility_score = sum(components) / len(components) if components else None

    return {
        "plausibility_score": plausibility_score,
        "avg_basket_prior_score": avg_prior,
        "avg_query_overlap_score": avg_overlap,
        "avg_preference_hits": avg_pref,
        "avg_processed_penalty": avg_processed,
        "pairwise_name_diversity": diversity,
    }


def compute_retrieval_quality_metrics(candidates_df: pd.DataFrame, result: dict) -> dict:
    if candidates_df is None or candidates_df.empty:
        return {
            "retrieval_top5_relevance_proxy": None,
            "retrieval_junk_rate": None,
            "retrieval_duplicate_rate": None,
            "retrieval_candidate_count": 0,
        }

    item_col = get_item_name_col(candidates_df)
    score_col = get_score_col(candidates_df)
    overlap_col = get_overlap_col(candidates_df)
    pref_col = get_prefhit_col(candidates_df)

    junk_terms = [
        "baby food", "formula", "cat food", "dog food", "snack", "candy", "dessert",
        "cookie", "chips", "soda", "cola"
    ]

    work = candidates_df.copy()

    if score_col:
        work = work.sort_values(score_col, ascending=False)
    elif overlap_col:
        work = work.sort_values(overlap_col, ascending=False)

    top5 = work.head(5).copy()

    # Relevance proxy based on overlap + preference hits
    rel_parts = []
    if overlap_col and overlap_col in top5.columns:
        rel_parts.append(top5[overlap_col].fillna(0).mean())
    if pref_col and pref_col in top5.columns:
        rel_parts.append(top5[pref_col].fillna(0).mean())
    relevance_proxy = sum(rel_parts) / len(rel_parts) if rel_parts else None

    junk_rate = None
    duplicate_rate = None

    if item_col:
        names = work[item_col].fillna("").astype(str).tolist()
        lowered = [normalize_text(x) for x in names]
        if lowered:
            junk_count = sum(any(j in x for j in junk_terms) for x in lowered)
            junk_rate = junk_count / len(lowered)

            unique_count = len(set(lowered))
            duplicate_rate = 1 - (unique_count / len(lowered))

    return {
        "retrieval_top5_relevance_proxy": relevance_proxy,
        "retrieval_junk_rate": junk_rate,
        "retrieval_duplicate_rate": duplicate_rate,
        "retrieval_candidate_count": len(candidates_df),
    }


def build_pipeline(instacart_bundle, max_prior_orders: int) -> GroceryOptimizationPipelineLive:
    return GroceryOptimizationPipelineLive(
        instacart_bundle=instacart_bundle,
        usda_api_key=settings.usda_api_key,
        max_prior_orders=max_prior_orders,
    )


def disable_priors_on_pipeline(pipeline):
    if hasattr(pipeline, "prior_builder"):
        pipeline.prior_builder = None
    if hasattr(pipeline, "retriever") and hasattr(pipeline.retriever, "prior_builder"):
        pipeline.retriever.prior_builder = None
    return pipeline


def maybe_disable_llm_explanation(result: dict, enabled: bool):
    if enabled:
        return result
    if "explanation" not in result or result["explanation"] is None:
        result["explanation"] = "LLM explanation disabled."
    return result


def build_naive_basket_from_candidates(candidates_df: pd.DataFrame, constraints: dict, basket_size: int = 5):
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame()

    work = candidates_df.copy()

    score_col = get_score_col(work)
    overlap_col = get_overlap_col(work)
    price_col = get_price_col(work)
    protein_col = get_protein_col(work)

    sort_cols = []
    ascending = []

    if overlap_col:
        sort_cols.append(overlap_col)
        ascending.append(False)
    if protein_col:
        sort_cols.append(protein_col)
        ascending.append(False)
    if price_col:
        sort_cols.append(price_col)
        ascending.append(True)
    if score_col:
        sort_cols.insert(0, score_col)
        ascending.insert(0, False)

    if sort_cols:
        work = work.sort_values(sort_cols, ascending=ascending)

    budget = constraints.get("budget")
    avoid_items = constraints.get("avoid_items", []) or []
    item_col = get_item_name_col(work)
    price_col = get_price_col(work)

    chosen_rows = []
    total_cost = 0.0

    for _, row in work.iterrows():
        row_name = normalize_text(row[item_col]) if item_col else ""
        if any(normalize_text(a) in row_name for a in avoid_items):
            continue

        row_price = safe_num(row[price_col], 0.0) if price_col else 0.0
        if budget is not None and (total_cost + row_price > budget):
            continue

        chosen_rows.append(row)
        total_cost += row_price

        if len(chosen_rows) >= basket_size:
            break

    if not chosen_rows:
        return pd.DataFrame(columns=work.columns)

    return pd.DataFrame(chosen_rows).reset_index(drop=True)


def summarize_basket_df(basket_df: pd.DataFrame):
    if basket_df is None or basket_df.empty:
        return {
            "total_cost": 0.0,
            "num_items": 0,
            "total_protein_g": 0.0,
            "total_calories": 0.0,
        }

    price_col = get_price_col(basket_df)
    protein_col = get_protein_col(basket_df)
    calorie_col = get_calorie_col(basket_df)

    return {
        "total_cost": float(basket_df[price_col].fillna(0).sum()) if price_col else 0.0,
        "num_items": int(len(basket_df)),
        "total_protein_g": float(basket_df[protein_col].fillna(0).sum()) if protein_col else 0.0,
        "total_calories": float(basket_df[calorie_col].fillna(0).sum()) if calorie_col else 0.0,
    }


def run_pipeline_once(
    user_query: str,
    use_priors: bool,
    use_llm: bool,
    top_k: int,
    max_prior_orders: int,
    instacart_bundle,
):
    pipeline = build_pipeline(instacart_bundle=instacart_bundle, max_prior_orders=max_prior_orders)

    if not use_priors:
        pipeline = disable_priors_on_pipeline(pipeline)

    start = time.time()
    result = pipeline.run(
        user_query=user_query,
        use_llm=use_llm,
        top_k=top_k,
    )
    total_runtime = time.time() - start

    result = maybe_disable_llm_explanation(result, enabled=use_llm)

    timing = result.get("timing", {})
    if not timing:
        timing = {
            "parse_time": None,
            "plan_time": None,
            "retrieve_time": None,
            "optimize_time": None,
            "explain_time": None,
            "total_time": round(total_runtime, 4),
        }
    elif timing.get("total_time") is None:
        timing["total_time"] = round(total_runtime, 4)

    candidates_df = result.get("candidates_df", pd.DataFrame())
    retrieval_stats = result.get("retrieval_stats", {})
    if not retrieval_stats:
        retrieval_stats = {
            "raw_candidates": len(candidates_df),
            "after_filters": len(candidates_df),
            "after_dedup": len(candidates_df),
            "final_candidates": len(candidates_df),
        }

    return result, timing, retrieval_stats


def run_variant(
    user_query: str,
    variant_name: str,
    variant_mode: str,
    use_priors: bool,
    use_llm: bool,
    top_k: int,
    max_prior_orders: int,
    instacart_bundle,
    naive_basket_size: int = 5,
):
    base_result, timing, retrieval_stats = run_pipeline_once(
        user_query=user_query,
        use_priors=use_priors,
        use_llm=use_llm,
        top_k=top_k,
        max_prior_orders=max_prior_orders,
        instacart_bundle=instacart_bundle,
    )

    result = base_result.copy()
    result["variant_name"] = variant_name
    result["variant_mode"] = variant_mode

    if variant_mode == "naive":
        naive_basket = build_naive_basket_from_candidates(
            candidates_df=result.get("candidates_df", pd.DataFrame()),
            constraints=result.get("constraints", {}),
            basket_size=naive_basket_size,
        )
        result["basket_df"] = naive_basket
        result["summary"] = summarize_basket_df(naive_basket)
        result["optimization_details"] = {
            "mode": "naive_top_candidates",
            "description": "No optimization; takes high-ranked candidates greedily under simple filtering."
        }

    candidates_df = result.get("candidates_df", pd.DataFrame())
    basket_df = result.get("basket_df", pd.DataFrame())

    constraint_metrics = compute_constraint_metrics(result)
    plausibility_metrics = compute_plausibility_metrics(basket_df)
    retrieval_quality_metrics = compute_retrieval_quality_metrics(candidates_df, result)

    return result, timing, retrieval_stats, constraint_metrics, plausibility_metrics, retrieval_quality_metrics


def save_single_run_outputs(run_id: str, result: dict):
    basket_path = SINGLE_RUN_DIR / f"{run_id}_basket.csv"
    candidate_path = SINGLE_RUN_DIR / f"{run_id}_candidates.csv"

    basket_df = result.get("basket_df", pd.DataFrame())
    candidates_df = result.get("candidates_df", pd.DataFrame())

    basket_df.to_csv(basket_path, index=False)
    candidates_df.to_csv(candidate_path, index=False)
    return basket_path, candidate_path


def save_reference_tables(top_k: int, max_prior_orders: int):
    final_config = pd.DataFrame([
        {"Parameter": "USDA candidates to fetch (top_k)", "Value": top_k},
        {"Parameter": "Max Instacart orders for prior", "Value": max_prior_orders},
        {"Parameter": "Primary system includes priors", "Value": "Yes"},
        {"Parameter": "LLM agent steps", "Value": "Optional"},
        {"Parameter": "Naive baseline", "Value": "Greedy top-candidate selection without optimization"},
        {"Parameter": "Optimizer", "Value": "Brute-force subset search"},
        {"Parameter": "Candidate cap before optimization", "Value": 20},
        {"Parameter": "Basket size range", "Value": "3-10"},
    ])
    final_config.to_csv(TABLES_DIR / "final_configuration.csv", index=False)

    scoring_table = pd.DataFrame([
        {"Component": "protein_g", "Weight": "+0.12", "Effect": "Rewards higher-protein foods"},
        {"Component": "fiber_g", "Weight": "+0.03", "Effect": "Rewards higher-fiber foods"},
        {"Component": "estimated_price", "Weight": "-0.02", "Effect": "Penalizes more expensive foods"},
        {"Component": "calories", "Weight": "-0.01", "Effect": "Slightly penalizes higher-calorie foods"},
        {"Component": "sugar_penalty", "Weight": "-0.04", "Effect": "Penalizes more sugar / added sugar"},
        {"Component": "processed_penalty", "Weight": "-0.20", "Effect": "Penalizes more processed foods"},
        {"Component": "preference_hits", "Weight": "+1.40", "Effect": "Strongly rewards foods matching user preferences"},
        {"Component": "query_overlap_score", "Weight": "+1.00", "Effect": "Rewards overlap with query terms"},
        {"Component": "basket_prior_score", "Weight": "+1.25", "Effect": "Rewards foods fitting historical Instacart basket patterns"},
    ])
    scoring_table.to_csv(TABLES_DIR / "scoring_function.csv", index=False)


def default_prompt_library():
    return [
        {
            "prompt_id": "P01",
            "prompt_type": "Direct",
            "difficulty": "Easy",
            "prompt_text": (
                "Build a high-protein weekly grocery basket for 5 dinners and 3 breakfasts under $70. "
                "Prefer chicken, eggs, greek yogurt, oats, rice, spinach, broccoli, bananas, and berries. "
                "Avoid shellfish. Keep added sugar low."
            ),
        },
        {
            "prompt_id": "P02",
            "prompt_type": "Vague",
            "difficulty": "Easy",
            "prompt_text": "I want affordable, healthy groceries that keep me full after exercise.",
        },
        {
            "prompt_id": "P03",
            "prompt_type": "Budget-tight",
            "difficulty": "Medium",
            "prompt_text": (
                "Give me a cheap grocery basket under $25 that can support 3 days of high-protein meals. "
                "Avoid sugary snacks."
            ),
        },
        {
            "prompt_id": "P04",
            "prompt_type": "Allergy",
            "difficulty": "Medium",
            "prompt_text": (
                "Build a healthy grocery basket for the week under $50. Avoid peanuts, shellfish, and dairy."
            ),
        },
        {
            "prompt_id": "P05",
            "prompt_type": "Vegetarian",
            "difficulty": "Medium",
            "prompt_text": (
                "I want a vegetarian grocery basket with lots of protein and fiber, low added sugar, under $60."
            ),
        },
        {
            "prompt_id": "P06",
            "prompt_type": "Conflicting goals",
            "difficulty": "Hard",
            "prompt_text": (
                "Build a very cheap grocery basket that is also high protein, low calorie, and low sugar."
            ),
        },
        {
            "prompt_id": "P07",
            "prompt_type": "Meal-style",
            "difficulty": "Medium",
            "prompt_text": (
                "Create a grocery basket for breakfasts and lunches that is healthy, filling, and easy to prep."
            ),
        },
        {
            "prompt_id": "P08",
            "prompt_type": "Athlete",
            "difficulty": "Medium",
            "prompt_text": (
                "I train 5 days a week. Give me groceries that support recovery, muscle gain, and convenience."
            ),
        },
        {
            "prompt_id": "P09",
            "prompt_type": "Low sugar",
            "difficulty": "Medium",
            "prompt_text": (
                "I want a low-sugar grocery basket under $45 with foods I can snack on during the day."
            ),
        },
        {
            "prompt_id": "P10",
            "prompt_type": "Family-ish",
            "difficulty": "Hard",
            "prompt_text": (
                "Build a grocery basket for several dinners that feels realistic, not random, and stays affordable."
            ),
        },
    ]


def save_prompt_library_table(prompt_rows):
    df = pd.DataFrame(prompt_rows)
    df.to_csv(TABLES_DIR / "evaluation_prompts.csv", index=False)
    return df


def generate_plots(prompt_results: pd.DataFrame, aggregate_results: pd.DataFrame, ablation_results: pd.DataFrame):
    if prompt_results.empty:
        return

    # Runtime by system
    runtime_pivot = prompt_results.groupby(["system"])["runtime_s"].mean().reset_index()
    ax = runtime_pivot.plot(kind="bar", x="system", y="runtime_s", figsize=(8, 5), legend=False)
    ax.set_title("Average Runtime by System")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xlabel("System")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "avg_runtime_by_system.png", dpi=300)
    plt.close()

    # Constraint satisfaction rate
    tmp = prompt_results.copy()
    tmp["constraint_success_num"] = tmp["overall_constraint_satisfaction"].map(
        {"Yes": 1, "No": 0, "NA": None}
    )
    success_df = tmp.groupby("system", dropna=False)["constraint_success_num"].mean().reset_index()
    ax = success_df.plot(kind="bar", x="system", y="constraint_success_num", figsize=(8, 5), legend=False)
    ax.set_title("Constraint Satisfaction Rate by System")
    ax.set_ylabel("Rate")
    ax.set_xlabel("System")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "constraint_satisfaction_rate.png", dpi=300)
    plt.close()

    # Plausibility by system
    plaus_df = prompt_results.groupby("system", dropna=False)["plausibility_score"].mean().reset_index()
    ax = plaus_df.plot(kind="bar", x="system", y="plausibility_score", figsize=(8, 5), legend=False)
    ax.set_title("Average Plausibility Score by System")
    ax.set_ylabel("Score")
    ax.set_xlabel("System")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plausibility_by_system.png", dpi=300)
    plt.close()

    # Retrieval relevance proxy
    rel_df = prompt_results.groupby("system", dropna=False)["retrieval_top5_relevance_proxy"].mean().reset_index()
    ax = rel_df.plot(kind="bar", x="system", y="retrieval_top5_relevance_proxy", figsize=(8, 5), legend=False)
    ax.set_title("Top-5 Retrieval Relevance Proxy by System")
    ax.set_ylabel("Score")
    ax.set_xlabel("System")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "retrieval_relevance_proxy.png", dpi=300)
    plt.close()

    # Runtime vs plausibility scatter
    scatter_df = prompt_results.dropna(subset=["runtime_s", "plausibility_score"]).copy()
    if not scatter_df.empty:
        plt.figure(figsize=(7, 5))
        for system in scatter_df["system"].unique():
            sub = scatter_df[scatter_df["system"] == system]
            plt.scatter(sub["runtime_s"], sub["plausibility_score"], label=system)
        plt.xlabel("Runtime (s)")
        plt.ylabel("Plausibility Score")
        plt.title("Runtime vs Plausibility")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "runtime_vs_plausibility.png", dpi=300)
        plt.close()

    # Parameter sensitivity
    if ablation_results is not None and not ablation_results.empty:
        sens = ablation_results.groupby(["ablation_name"])["runtime_s"].mean().reset_index()
        ax = sens.plot(kind="bar", x="ablation_name", y="runtime_s", figsize=(10, 5), legend=False)
        ax.set_title("Average Runtime in Sensitivity / Ablation Runs")
        ax.set_ylabel("Runtime (s)")
        ax.set_xlabel("Ablation")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "ablation_runtime.png", dpi=300)
        plt.close()


def build_system_comparison_table(prompt_results: pd.DataFrame) -> pd.DataFrame:
    if prompt_results.empty:
        return pd.DataFrame()

    rows = []
    for system in prompt_results["system"].unique():
        sub = prompt_results[prompt_results["system"] == system].copy()

        rows.append({
            "System": system,
            "Mean Runtime (s)": sub["runtime_s"].mean(),
            "Mean Cost": sub["total_cost"].mean(),
            "Mean Items": sub["num_items"].mean(),
            "Mean Protein (g)": sub["total_protein_g"].mean(),
            "Mean Calories": sub["total_calories"].mean(),
            "Constraint Satisfaction Rate": sub["constraint_success_num"].mean() if "constraint_success_num" in sub.columns else None,
            "Mean Plausibility Score": sub["plausibility_score"].mean(),
            "Mean Retrieval Relevance Proxy": sub["retrieval_top5_relevance_proxy"].mean(),
            "Mean Retrieval Junk Rate": sub["retrieval_junk_rate"].mean(),
            "Mean Retrieval Duplicate Rate": sub["retrieval_duplicate_rate"].mean(),
        })

    return pd.DataFrame(rows)


def build_failure_cases(prompt_results: pd.DataFrame, n_cases: int = 5) -> pd.DataFrame:
    if prompt_results.empty:
        return pd.DataFrame()

    work = prompt_results.copy()

    work["constraint_fail_num"] = work["overall_constraint_satisfaction"].map(
        {"Yes": 0, "No": 1, "NA": 0}
    )

    # Lower plausibility and higher runtime are worse
    work["plausibility_rank_val"] = work["plausibility_score"].fillna(-999)
    work["runtime_rank_val"] = work["runtime_s"].fillna(999)

    work = work.sort_values(
        by=["constraint_fail_num", "plausibility_rank_val", "runtime_rank_val"],
        ascending=[False, True, False]
    )

    cols = [
        "prompt_id", "prompt_type", "difficulty", "system", "runtime_s",
        "total_cost", "num_items", "total_protein_g", "total_calories",
        "overall_constraint_satisfaction", "plausibility_score",
        "retrieval_top5_relevance_proxy", "retrieval_junk_rate",
        "retrieval_duplicate_rate", "avoid_violation_terms", "prompt_text"
    ]
    cols = [c for c in cols if c in work.columns]
    return work[cols].head(n_cases)


def run_ablation_suite(
    prompt_rows,
    instacart_bundle,
    sensitivity_topk_values,
    max_prior_orders,
):
    ablation_rows = []

    settings_list = [
        {"ablation_name": "topk_5", "top_k": 5, "use_priors": True, "use_llm": False},
        {"ablation_name": "topk_10", "top_k": 10, "use_priors": True, "use_llm": False},
        {"ablation_name": "topk_20", "top_k": 20, "use_priors": True, "use_llm": False},
        {"ablation_name": "no_priors", "top_k": 10, "use_priors": False, "use_llm": False},
        {"ablation_name": "with_priors", "top_k": 10, "use_priors": True, "use_llm": False},
        {"ablation_name": "with_priors_llm", "top_k": 10, "use_priors": True, "use_llm": True},
    ]

    # Respect user-provided sensitivity_topk_values if passed
    custom_topks = sorted(set([int(x) for x in sensitivity_topk_values if str(x).strip().isdigit()]))
    if custom_topks:
        settings_list = [
            {"ablation_name": f"topk_{k}", "top_k": k, "use_priors": True, "use_llm": False}
            for k in custom_topks
        ] + [
            {"ablation_name": "no_priors", "top_k": 10, "use_priors": False, "use_llm": False},
            {"ablation_name": "with_priors", "top_k": 10, "use_priors": True, "use_llm": False},
            {"ablation_name": "with_priors_llm", "top_k": 10, "use_priors": True, "use_llm": True},
        ]

    for prompt in prompt_rows:
        for cfg in settings_list:
            result, timing, retrieval_stats, constraint_metrics, plausibility_metrics, retrieval_quality_metrics = run_variant(
                user_query=prompt["prompt_text"],
                variant_name=cfg["ablation_name"],
                variant_mode="optimized",
                use_priors=cfg["use_priors"],
                use_llm=cfg["use_llm"],
                top_k=cfg["top_k"],
                max_prior_orders=max_prior_orders,
                instacart_bundle=instacart_bundle,
            )

            ablation_rows.append({
                "prompt_id": prompt["prompt_id"],
                "prompt_type": prompt["prompt_type"],
                "difficulty": prompt["difficulty"],
                "ablation_name": cfg["ablation_name"],
                "top_k": cfg["top_k"],
                "use_priors": cfg["use_priors"],
                "use_llm": cfg["use_llm"],
                "runtime_s": timing.get("total_time", None),
                "total_cost": result["summary"]["total_cost"],
                "num_items": result["summary"]["num_items"],
                "total_protein_g": result["summary"]["total_protein_g"],
                "total_calories": result["summary"]["total_calories"],
                "overall_constraint_satisfaction": safe_yes_no(constraint_metrics["overall_constraint_satisfaction"]),
                "plausibility_score": plausibility_metrics["plausibility_score"],
                "retrieval_top5_relevance_proxy": retrieval_quality_metrics["retrieval_top5_relevance_proxy"],
            })

    return pd.DataFrame(ablation_rows)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    use_llm = st.checkbox("Use optional LLM agent steps", value=False)
    use_priors = st.checkbox("Use Instacart priors", value=True)
    top_k = st.slider("USDA candidates to fetch", 5, 30, 10, step=5)
    max_prior_orders = st.slider("Max Instacart orders for prior", 500, 5000, 1000, step=500)
    naive_basket_size = st.slider("Naive baseline basket size", 3, 10, 5, step=1)

    st.markdown("---")
    st.subheader("Single-run Prompts")
    direct_query = st.text_area(
        "Direct prompt",
        value=(
            "Build a high-protein weekly grocery basket for 5 dinners and 3 breakfasts under $70. "
            "Prefer chicken, eggs, greek yogurt, oats, rice, spinach, broccoli, bananas, and berries. "
            "Avoid shellfish. Keep added sugar low."
        ),
        height=120,
    )
    vague_query = st.text_area(
        "Vague prompt",
        value="I want affordable, healthy groceries that keep me full after exercise.",
        height=90,
    )

    st.markdown("---")
    st.subheader("Evaluation Suite")
    prompt_library_default = json.dumps(default_prompt_library(), indent=2)
    prompt_library_text = st.text_area(
        "Prompt library (JSON list)",
        value=prompt_library_default,
        height=340,
    )

    sensitivity_topk = st.text_input(
        "Sensitivity top_k values (comma-separated)",
        value="5,10,20"
    )

query = st.text_area("Describe your shopping goal", value=direct_query, height=150)

instacart_ok = instacart_data_available(settings.instacart_dir)
st.info(f"USDA API key loaded: {'Yes' if bool(settings.usda_api_key) else 'No'}")
st.info(f"Instacart files detected: {'Yes' if instacart_ok else 'No'}")
st.info(f"OpenAI API key loaded: {'Yes' if bool(settings.openai_api_key) else 'No'}")

# -----------------------------------------------------------------------------
# Load Instacart once
# -----------------------------------------------------------------------------
if instacart_ok:
    instacart_bundle = cached_load_instacart(settings.instacart_dir)
else:
    instacart_bundle = None

save_reference_tables(top_k=top_k, max_prior_orders=max_prior_orders)

# -----------------------------------------------------------------------------
# Single run
# -----------------------------------------------------------------------------
if st.button("Run optimizer", type="primary"):
    try:
        result, timing, retrieval_stats, constraint_metrics, plausibility_metrics, retrieval_quality_metrics = run_variant(
            user_query=query,
            variant_name="Custom Run",
            variant_mode="optimized",
            use_priors=use_priors,
            use_llm=use_llm,
            top_k=top_k,
            max_prior_orders=max_prior_orders,
            instacart_bundle=instacart_bundle,
            naive_basket_size=naive_basket_size,
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        basket_path, candidate_path = save_single_run_outputs(run_id, result)

        st.subheader("Recommended basket")
        st.dataframe(result["basket_df"], width="stretch")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estimated total", f"${result['summary']['total_cost']:.2f}")
        c2.metric("Items", int(result["summary"]["num_items"]))
        c3.metric("Protein", f"{result['summary']['total_protein_g']:.1f} g")
        c4.metric("Calories", f"{result['summary']['total_calories']:.0f}")

        st.subheader("Explanation")
        st.write(result.get("explanation", ""))

        summary_table = pd.DataFrame([
            {"Metric": "Runtime (s)", "Value": timing.get("total_time", "")},
            {"Metric": "Budget met", "Value": safe_yes_no(constraint_metrics["budget_met"])},
            {"Metric": "Budget margin", "Value": constraint_metrics["budget_margin"]},
            {"Metric": "Exclusions met", "Value": safe_yes_no(constraint_metrics["exclusions_met"])},
            {"Metric": "Protein target met", "Value": safe_yes_no(constraint_metrics["protein_target_met"])},
            {"Metric": "Protein margin (g)", "Value": constraint_metrics["protein_margin_g"]},
            {"Metric": "Overall constraint satisfaction", "Value": safe_yes_no(constraint_metrics["overall_constraint_satisfaction"])},
            {"Metric": "Plausibility score", "Value": plausibility_metrics["plausibility_score"]},
            {"Metric": "Avg basket prior score", "Value": plausibility_metrics["avg_basket_prior_score"]},
            {"Metric": "Top-5 retrieval relevance proxy", "Value": retrieval_quality_metrics["retrieval_top5_relevance_proxy"]},
            {"Metric": "Retrieval junk rate", "Value": retrieval_quality_metrics["retrieval_junk_rate"]},
            {"Metric": "Retrieval duplicate rate", "Value": retrieval_quality_metrics["retrieval_duplicate_rate"]},
            {"Metric": "Raw candidates", "Value": retrieval_stats.get("raw_candidates", len(result["candidates_df"]))},
            {"Metric": "Final candidates", "Value": retrieval_stats.get("final_candidates", len(result["candidates_df"]))},
            {"Metric": "Basket CSV", "Value": str(basket_path)},
            {"Metric": "Candidates CSV", "Value": str(candidate_path)},
        ])
        summary_table["Value"] = summary_table["Value"].astype(str)

        st.subheader("Run Summary")
        st.dataframe(summary_table, width="stretch")

        with st.expander("Parsed constraints", expanded=False):
            st.json(result.get("constraints", {}))

        with st.expander("Plan", expanded=False):
            st.json(result.get("plan", {}))

        with st.expander("Candidate items", expanded=False):
            st.dataframe(result.get("candidates_df", pd.DataFrame()), width="stretch")

        with st.expander("Optimization details", expanded=False):
            st.json(result.get("optimization_details", {}))

    except Exception as e:
        st.error(f"Error while running optimizer: {e}")

# -----------------------------------------------------------------------------
# Comparison suite
# -----------------------------------------------------------------------------
if st.button("Run comparison suite"):
    try:
        variants = [
            {"system": "Naive Baseline", "variant_mode": "naive", "use_priors": False, "use_llm": False},
            {"system": "Baseline", "variant_mode": "optimized", "use_priors": False, "use_llm": False},
            {"system": "Priors", "variant_mode": "optimized", "use_priors": True, "use_llm": False},
            {"system": "Priors + LLM", "variant_mode": "optimized", "use_priors": True, "use_llm": True},
        ]

        try:
            prompt_rows = json.loads(prompt_library_text)
            if not isinstance(prompt_rows, list):
                raise ValueError("Prompt library must be a JSON list.")
        except Exception as e:
            st.error(f"Prompt library JSON is invalid: {e}")
            st.stop()

        prompt_df = save_prompt_library_table(prompt_rows)
        sensitivity_topk_values = [x.strip() for x in sensitivity_topk.split(",") if x.strip()]

        rows = []

        with st.spinner("Running comparison suite and generating outputs..."):
            for prompt in prompt_rows:
                for variant in variants:
                    result, timing, retrieval_stats, constraint_metrics, plausibility_metrics, retrieval_quality_metrics = run_variant(
                        user_query=prompt["prompt_text"],
                        variant_name=variant["system"],
                        variant_mode=variant["variant_mode"],
                        use_priors=variant["use_priors"],
                        use_llm=variant["use_llm"],
                        top_k=top_k,
                        max_prior_orders=max_prior_orders,
                        instacart_bundle=instacart_bundle,
                        naive_basket_size=naive_basket_size,
                    )

                    rows.append({
                        "prompt_id": prompt.get("prompt_id", ""),
                        "prompt_type": prompt.get("prompt_type", ""),
                        "difficulty": prompt.get("difficulty", ""),
                        "prompt_text": prompt.get("prompt_text", ""),
                        "system": variant["system"],
                        "variant_mode": variant["variant_mode"],
                        "use_priors": variant["use_priors"],
                        "use_llm": variant["use_llm"],
                        "runtime_s": timing.get("total_time", None),
                        "total_cost": result["summary"]["total_cost"],
                        "num_items": result["summary"]["num_items"],
                        "total_protein_g": result["summary"]["total_protein_g"],
                        "total_calories": result["summary"]["total_calories"],
                        "budget_met": safe_yes_no(constraint_metrics["budget_met"]),
                        "exclusions_met": safe_yes_no(constraint_metrics["exclusions_met"]),
                        "protein_target_met": safe_yes_no(constraint_metrics["protein_target_met"]),
                        "overall_constraint_satisfaction": safe_yes_no(constraint_metrics["overall_constraint_satisfaction"]),
                        "budget_margin": constraint_metrics["budget_margin"],
                        "protein_margin_g": constraint_metrics["protein_margin_g"],
                        "preferred_hit_rate": constraint_metrics["preferred_hit_rate"],
                        "avoid_violation_terms": constraint_metrics["avoid_violation_terms"],
                        "raw_candidates": retrieval_stats.get("raw_candidates", len(result["candidates_df"])),
                        "final_candidates": retrieval_stats.get("final_candidates", len(result["candidates_df"])),
                        "plausibility_score": plausibility_metrics["plausibility_score"],
                        "avg_basket_prior_score": plausibility_metrics["avg_basket_prior_score"],
                        "avg_query_overlap_score": plausibility_metrics["avg_query_overlap_score"],
                        "avg_preference_hits": plausibility_metrics["avg_preference_hits"],
                        "avg_processed_penalty": plausibility_metrics["avg_processed_penalty"],
                        "pairwise_name_diversity": plausibility_metrics["pairwise_name_diversity"],
                        "retrieval_top5_relevance_proxy": retrieval_quality_metrics["retrieval_top5_relevance_proxy"],
                        "retrieval_junk_rate": retrieval_quality_metrics["retrieval_junk_rate"],
                        "retrieval_duplicate_rate": retrieval_quality_metrics["retrieval_duplicate_rate"],
                    })

        prompt_results = pd.DataFrame(rows)

        prompt_results["constraint_success_num"] = prompt_results["overall_constraint_satisfaction"].map(
            {"Yes": 1, "No": 0, "NA": None}
        )

        prompt_results.to_csv(TABLES_DIR / "prompt_level_results.csv", index=False)

        system_comparison = build_system_comparison_table(prompt_results)
        system_comparison.to_csv(TABLES_DIR / "system_comparison.csv", index=False)

        failure_cases = build_failure_cases(prompt_results, n_cases=8)
        failure_cases.to_csv(FAILURE_DIR / "failure_cases.csv", index=False)

        ablation_results = run_ablation_suite(
            prompt_rows=prompt_rows,
            instacart_bundle=instacart_bundle,
            sensitivity_topk_values=sensitivity_topk_values,
            max_prior_orders=max_prior_orders,
        )
        ablation_results.to_csv(TABLES_DIR / "ablation_results.csv", index=False)

        generate_plots(prompt_results, system_comparison, ablation_results)

        st.success(f"Saved prompt-level results to {TABLES_DIR / 'prompt_level_results.csv'}")
        st.success(f"Saved system comparison table to {TABLES_DIR / 'system_comparison.csv'}")
        st.success(f"Saved failure cases to {FAILURE_DIR / 'failure_cases.csv'}")
        st.success(f"Saved ablation results to {TABLES_DIR / 'ablation_results.csv'}")
        st.success(f"Saved plots to {PLOTS_DIR}")

        st.subheader("Evaluation Prompt Library")
        st.dataframe(prompt_df, width="stretch")

        st.subheader("Prompt-level Results")
        st.dataframe(prompt_results, width="stretch")

        st.subheader("System Comparison")
        st.dataframe(system_comparison, width="stretch")

        st.subheader("Failure Cases")
        st.dataframe(failure_cases, width="stretch")

        st.subheader("Ablation / Sensitivity Results")
        st.dataframe(ablation_results, width="stretch")

        st.subheader("Saved Plot Files")
        plot_files = pd.DataFrame([
            {"Plot": "Average runtime by system", "Path": str(PLOTS_DIR / "avg_runtime_by_system.png")},
            {"Plot": "Constraint satisfaction rate", "Path": str(PLOTS_DIR / "constraint_satisfaction_rate.png")},
            {"Plot": "Plausibility by system", "Path": str(PLOTS_DIR / "plausibility_by_system.png")},
            {"Plot": "Retrieval relevance proxy", "Path": str(PLOTS_DIR / "retrieval_relevance_proxy.png")},
            {"Plot": "Runtime vs plausibility", "Path": str(PLOTS_DIR / "runtime_vs_plausibility.png")},
            {"Plot": "Ablation runtime", "Path": str(PLOTS_DIR / "ablation_runtime.png")},
        ])
        st.dataframe(plot_files, width="stretch")

    except Exception as e:
        st.error(f"Error while running comparison suite: {e}")

print("complete")