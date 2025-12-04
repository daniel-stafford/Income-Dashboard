import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. PAGE SETUP
st.set_page_config(layout="wide", page_title="Income Dashboard")


# 2. LOAD DATA
@st.cache_data
def load_data():
    # Looks for file in the same folder
    return pd.read_csv("dashboard_data.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Error: 'dashboard_data.csv' not found. Make sure it is in the same folder as app.py"
    )
    st.stop()

# Prepare Sorted List for Dropdown
survey_scores = (
    df[["cy", "r2_score"]].drop_duplicates().sort_values(by="r2_score", ascending=False)
)
survey_dict = dict(zip(survey_scores["cy"], survey_scores["r2_score"]))

# 3. SIDEBAR SELECTION
st.sidebar.header("Settings")
selected_cy = st.sidebar.selectbox(
    "Select Survey (Best to Worst Fit):",
    options=survey_scores["cy"],
    format_func=lambda x: f"{x} (R2: {survey_dict[x]:.3f})",
)

# 4. FILTER DATA
dff = df[df["cy"] == selected_cy].copy()

# Reshape for Plotly
df_long = dff.melt(
    id_vars=["cy", "percentile", "true_daily_consumption", "r2_score"],
    value_vars=[c for c in df.columns if c.startswith("pred_")],
    var_name="Model_Code",
    value_name="Prediction",
)

model_map = {
    "pred_m1": "1. Global Base",
    "pred_m2": "2. Gini Interaction",
    "pred_m3": "3. Country FE",
    "pred_m4": "4. Survey Linear",
    "pred_m5": "5. Survey Quad",
    "pred_m6": "6. Survey Cubic",
    "pred_m7": "7. Survey Quartic",
}
df_long["Model"] = df_long["Model_Code"].map(model_map)

# Add Truth Data
df_true = dff[["percentile", "true_daily_consumption"]].copy()
df_true["Prediction"] = df_true["true_daily_consumption"]
df_true["Model"] = "TRUE INCOME"

df_final = pd.concat([df_long, df_true], ignore_index=True)

# 5. PLOT
st.title(f"Survey: {selected_cy}")
st.caption(f"Model 7 R-Squared Score: {dff['r2_score'].iloc[0]:.4f}")

fig = px.line(
    df_final,
    x="percentile",
    y="Prediction",
    color="Model",
    color_discrete_map={
        "TRUE INCOME": "#F1C40F",
        "1. Global Base": "#440154",
        "2. Gini Interaction": "#482878",
        "3. Country FE": "#3e4989",
        "4. Survey Linear": "#31688e",
        "5. Survey Quad": "#26828e",
        "6. Survey Cubic": "#1f9e89",
        "7. Survey Quartic": "#35b779",
    },
    height=600,
)

fig.update_traces(
    mode="markers", marker=dict(size=6, opacity=0.8), selector=dict(name="TRUE INCOME")
)
fig.update_layout(
    template="plotly_white",
    legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
)

st.plotly_chart(fig, use_container_width=True)
