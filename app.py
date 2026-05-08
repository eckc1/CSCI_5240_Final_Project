import streamlit as st
from src.config import settings
from src.instacart_loader import load_instacart_data, instacart_data_available
from src.pipeline_live import GroceryOptimizationPipelineLive

st.set_page_config(page_title="Live Grocery Optimizer", layout="wide")

st.title("Grocery Store Optimizer — USDA + Instacart")
st.caption("Agentic grocery optimization using live USDA nutrition search and Instacart basket priors.")


@st.cache_data(show_spinner=True)
def cached_load_instacart(instacart_dir: str):
    return load_instacart_data(instacart_dir)


@st.cache_resource(show_spinner=True)
def cached_build_pipeline(instacart_bundle, usda_api_key: str | None, max_prior_orders: int):
    return GroceryOptimizationPipelineLive(
        instacart_bundle=instacart_bundle,
        usda_api_key=usda_api_key,
        max_prior_orders=max_prior_orders,
    )


with st.sidebar:
    st.header("Configuration")
    use_llm = st.checkbox("Use optional LLM agent steps", value=False)
    top_k = st.slider("USDA candidates to fetch", 5, 30, 10, step=5)
    max_prior_orders = st.slider("Max Instacart orders for prior", 500, 5000, 1000, step=500)

default_query = (
    "Build a high-protein weekly grocery basket for 5 dinners and 3 breakfasts under $70. "
    "Prefer chicken, eggs, greek yogurt, oats, rice, spinach, broccoli, bananas, and berries. "
    "Avoid shellfish. Keep added sugar low."
)

query = st.text_area("Describe your shopping goal", value=default_query, height=150)

instacart_ok = instacart_data_available(settings.instacart_dir)
st.info(f"USDA API key loaded: {'Yes' if bool(settings.usda_api_key) else 'No'}")
st.info(f"Instacart files detected: {'Yes' if instacart_ok else 'No'}")
st.info(f"OpenAI API key loaded: {'Yes' if bool(settings.openai_api_key) else 'No'}")

if st.button("Run optimizer", type="primary"):
    try:
        if instacart_ok:
            instacart_bundle = cached_load_instacart(settings.instacart_dir)
        else:
            instacart_bundle = None

        pipeline = cached_build_pipeline(
            instacart_bundle=instacart_bundle,
            usda_api_key=settings.usda_api_key,
            max_prior_orders=max_prior_orders,
        )

        with st.spinner("Running grocery optimization..."):
            result = pipeline.run(
                user_query=query,
                use_llm=use_llm,
                top_k=top_k,
            )

        st.subheader("Recommended basket")
        #st.dataframe(result["basket_df"], use_container_width=True)
        st.dataframe(result["basket_df"], width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estimated total", f"${result['summary']['total_cost']:.2f}")
        c2.metric("Items", int(result["summary"]["num_items"]))
        c3.metric("Protein", f"{result['summary']['total_protein_g']:.1f} g")
        c4.metric("Calories", f"{result['summary']['total_calories']:.0f}")

        st.subheader("Explanation")
        st.write(result["explanation"])

        with st.expander("Parsed constraints", expanded=False):
            st.json(result["constraints"])

        with st.expander("Plan", expanded=False):
            st.json(result["plan"])

        with st.expander("Candidate items", expanded=False):
            st.dataframe(result["candidates_df"], width="stretch")

        with st.expander("Optimization details", expanded=False):
            st.json(result["optimization_details"])

    except Exception as e:
        st.error(f"Error while running optimizer: {e}")