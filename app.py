import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. PAGE SETUP
st.set_page_config(layout="wide", page_title="Income Distribution Dashboard")

# --- CONFIGURATION MAPPING ---
# UPDATED: Renamed "Simulated" to "0. Simulated"
MODEL_CONFIG = {
    "r2_simulated": {"pred_col": "simulated_daily_consumption", "name": "0. Simulated"},
    "r2_m1_global": {"pred_col": "pred_m1_global", "name": "1. Global Base"},
    "r2_m2_gini": {"pred_col": "pred_m2_gini", "name": "2. Gini Interaction"},
    "r2_m3_cntry": {"pred_col": "pred_m3_cntry", "name": "3. Country FE"},
    "r2_m4_linear": {"pred_col": "pred_m4_linear", "name": "4. Simple Linear"},
    "r2_m5_quad": {"pred_col": "pred_m5_quadratic", "name": "5. Simple Quad"},
    "r2_m6_cubic": {"pred_col": "pred_m6_cubic", "name": "6. Simple Cubic"},
    "r2_m7_quartic": {"pred_col": "pred_m7_quartic", "name": "7. Simple Quartic"},
}


# 2. LOAD DATA
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dashboard_data.csv", sep=None, engine="python")
        df.columns = df.columns.str.strip().str.lower()
        if "iso3" not in df.columns:
            st.error("❌ CRITICAL ERROR: 'iso3' column not found.")
            st.stop()
        df["survey_id"] = df["iso3"] + "_" + df["year"].astype(str)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


raw_df = load_data()

if raw_df is None:
    st.stop()


# 3. STATS CALCULATION FUNCTIONS
def calculate_r2_manual(y_true, y_pred):
    """
    Robust R2 calculation that handles NaN alignment and zero variance.
    """
    try:
        # Ensure we are working with aligned numpy arrays
        y_t = np.array(y_true)
        y_p = np.array(y_pred)

        # Create a mask to only keep indices where BOTH true and pred exist
        mask = ~np.isnan(y_t) & ~np.isnan(y_p)
        y_t = y_t[mask]
        y_p = y_p[mask]

        if len(y_t) < 2:
            return -999.0  # Not enough data

        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)

        # Avoid division by zero if the data is perfectly flat
        if ss_tot < 1e-9:
            return 0.0  # Technically undefined, but 0 is safe for 'no predictive power'

        return 1 - (ss_res / ss_tot)
    except:
        return -999.0


@st.cache_data
def calculate_dynamic_stats(data_subset):
    results = []
    # Group by survey
    for survey_id, group in data_subset.groupby("survey_id"):
        row = {"survey_id": survey_id}
        # Use simple array access to avoid index alignment issues
        y_true = group["true_daily_consumption"].values

        for model_key, config in MODEL_CONFIG.items():
            if config["pred_col"] in group.columns:
                y_pred = group[config["pred_col"]].values
                score = calculate_r2_manual(y_true, y_pred)
                row[model_key] = score
            else:
                row[model_key] = -999.0
        results.append(row)
    return pd.DataFrame(results)


# 4. SIDEBAR CONTROLS
st.sidebar.markdown("## Settings")

# --- Range Slider ---
min_p, max_p = st.sidebar.slider(
    "Select Percentile Range", min_value=0, max_value=100, value=(0, 100), step=1
)
st.sidebar.caption(f"Analyzing percentiles: **{min_p}% to {max_p}%**")

# --- Tip ---
st.sidebar.info("💡 **Tip:** Try a range like **0-99** to filter out extreme outliers.")

# --- Filter Data ---
filtered_df = raw_df[
    (raw_df["percentile"] >= min_p) & (raw_df["percentile"] <= max_p)
].copy()

if filtered_df.empty:
    st.error(f"❌ No data found for percentiles {min_p}-{max_p}.")
    st.stop()

# --- Calculate Filtered Stats (Per Survey) ---
survey_stats = calculate_dynamic_stats(filtered_df)

# --- CALCULATE GLOBAL AVERAGES (Dynamic based on filter) ---
r2_cols = list(MODEL_CONFIG.keys())
stats_for_avg = survey_stats[r2_cols].copy()
# Replace failure code -999.0 with NaN so it doesn't skew average
stats_for_avg = stats_for_avg.replace(-999.0, np.nan)

global_means = stats_for_avg.mean().reset_index()
global_means.columns = ["model_key", "mean_r2"]
global_means["Model"] = global_means["model_key"].map(lambda x: MODEL_CONFIG[x]["name"])
global_means = global_means.sort_values(by="mean_r2", ascending=False)

# --- Determine Winners Per Survey ---
survey_stats["best_r2"] = survey_stats[r2_cols].max(axis=1)
survey_stats["winning_col"] = survey_stats[r2_cols].idxmax(axis=1)
survey_stats["winning_model"] = survey_stats["winning_col"].map(
    lambda x: MODEL_CONFIG[x]["name"] if pd.notnull(x) else "None"
)
# Filter out failures for display
survey_stats = survey_stats[survey_stats["best_r2"] > -100]
survey_stats = survey_stats.sort_values(by="best_r2", ascending=False)

# Lookup maps
best_score_map = dict(zip(survey_stats["survey_id"], survey_stats["best_r2"]))
sorted_survey_list = survey_stats["survey_id"].tolist()

if not sorted_survey_list:
    st.warning("Not enough data points to calculate R².")
    st.stop()

# --- PERSISTENT SURVEY SELECTION LOGIC ---
if "last_selected_survey" not in st.session_state:
    st.session_state.last_selected_survey = sorted_survey_list[0]


def update_survey_selection():
    st.session_state.last_selected_survey = st.session_state.survey_widget


try:
    pre_selected_index = sorted_survey_list.index(st.session_state.last_selected_survey)
except ValueError:
    pre_selected_index = 0


def format_func(survey_id):
    score = best_score_map.get(survey_id, 0)
    return f"{survey_id} (R²: {score:.2f})"


selected_survey = st.sidebar.selectbox(
    "Select Survey",
    options=sorted_survey_list,
    format_func=format_func,
    index=pre_selected_index,
    key="survey_widget",
    on_change=update_survey_selection,
)

# --- SIDEBAR: GLOBAL AVERAGES ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 Global Average R²")
st.sidebar.caption(
    f"Average performance across all surveys for **percentiles {min_p}-{max_p}**."
)

st.sidebar.dataframe(
    global_means[["Model", "mean_r2"]]
    .style.background_gradient(subset=["mean_r2"], cmap="Greens")
    .format({"mean_r2": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

# 5. MAIN CONTENT
st.title(f"Analysis: Percentiles {min_p} - {max_p}")

# --- UPDATED TAB LOGIC FOR PERSISTENCE ---
tab_options = ["📈 Chart Visualization", "📊 Model Analysis"]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = tab_options[0]

# Hidden logic: Using radio button as tabs
selected_tab = st.radio(
    "Select View",
    options=tab_options,
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",  # This binds the selection to session_state
)
st.markdown("---")  # Visual separator

# --- VIEW 1: CHART ---
if selected_tab == "📈 Chart Visualization":
    dff = filtered_df[filtered_df["survey_id"] == selected_survey].copy()
    survey_row = survey_stats[survey_stats["survey_id"] == selected_survey].iloc[0]

    st.markdown(f"### Survey: {selected_survey}")

    with st.expander(f"Show R² Scores (Percentiles {min_p}-{max_p})", expanded=True):
        score_data = []
        for r2_col in r2_cols:
            score = survey_row[r2_col]
            score_data.append(
                {"Model Name": MODEL_CONFIG[r2_col]["name"], "R² Score": score}
            )

        score_df = pd.DataFrame(score_data).sort_values(by="R² Score", ascending=False)
        st.dataframe(
            score_df.style.background_gradient(
                subset=["R² Score"], cmap="Greens"
            ).format({"R² Score": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # Chart
    rename_map = {"true_daily_consumption": "TRUE INCOME"}
    for r2_col, config in MODEL_CONFIG.items():
        if config["pred_col"] in dff.columns:
            rename_map[config["pred_col"]] = config["name"]

    plot_df = dff.rename(columns=rename_map).melt(
        id_vars=["percentile"],
        value_vars=list(rename_map.values()),
        var_name="Model",
        value_name="Prediction",
    )
    plot_df["Prediction"] = plot_df["Prediction"].round(2)

    color_map = {}
    available_colors = px.colors.qualitative.G10
    idx = 0
    for m in plot_df["Model"].unique():
        if m == "TRUE INCOME":
            color_map[m] = "#FFC107"
        elif m == "0. Simulated":  # UPDATED COLOR KEY
            color_map[m] = "#FF0000"
        else:
            color_map[m] = available_colors[idx % len(available_colors)]
            idx += 1

    fig = px.line(
        plot_df,
        x="percentile",
        y="Prediction",
        color="Model",
        color_discrete_map=color_map,
        height=600,
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_traces(yhoverformat=".2f")

    fig.update_traces(
        selector={"name": "TRUE INCOME"},
        mode="markers",
        marker=dict(size=8, color="#FFC107", symbol="circle"),
    )
    # UPDATED TRACE SELECTOR
    if "0. Simulated" in color_map:
        fig.update_traces(
            selector={"name": "0. Simulated"},
            line=dict(width=3, color="red", dash="dash"),
        )

    st.plotly_chart(fig, use_container_width=True)

# --- VIEW 2: MODEL ANALYSIS ---
elif selected_tab == "📊 Model Analysis":

    # 1. DISTRIBUTION CHART
    st.markdown(f"### Winning Models Distribution (Percentiles {min_p}-{max_p})")

    win_counts = survey_stats["winning_model"].value_counts().reset_index()
    win_counts.columns = ["Model", "Count"]
    win_counts["Percentage"] = (win_counts["Count"] / win_counts["Count"].sum()) * 100

    fig_bar = px.bar(
        win_counts,
        x="Model",
        y="Percentage",
        text_auto=".2f",
        color="Model",
        hover_data=["Count"],
    )
    fig_bar.update_layout(yaxis_title="Percentage Won (%)", showlegend=False)
    fig_bar.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")  # Divider

    # 2. LEADERBOARD TABLE
    st.markdown(f"### Best Models Leaderboard")
    st.caption(
        "Sorted list of the best performing model for every survey in this range."
    )
    st.dataframe(
        survey_stats[["survey_id", "winning_model", "best_r2"]]
        .style.background_gradient(subset=["best_r2"], cmap="Greens")
        .format({"best_r2": "{:.2f}"}),
        use_container_width=True,
    )
