import streamlit as st
import pandas as pd
import plotly.express as px

# 1. PAGE SETUP
st.set_page_config(layout="wide", page_title="Income Distribution Dashboard")

# --- CONFIGURATION MAPPING ---
MODEL_CONFIG = {
    "r2_simulated": {"pred_col": "simulated_daily_consumption", "name": "Simulated"},
    "r2_m1_global": {"pred_col": "pred_m1_global", "name": "1. Global Base"},
    "r2_m2_gini": {"pred_col": "pred_m2_gini", "name": "2. Gini Interaction"},
    "r2_m3_cntry": {"pred_col": "pred_m3_cntry", "name": "3. Country FE"},
    "r2_m4_linear": {"pred_col": "pred_m4_linear", "name": "4. Survey Linear"},
    "r2_m5_quad": {"pred_col": "pred_m5_quadratic", "name": "5. Survey Quad"},
    "r2_m6_cubic": {"pred_col": "pred_m6_cubic", "name": "6. Survey Cubic"},
    "r2_m7_quartic": {"pred_col": "pred_m7_quartic", "name": "7. Survey Quartic"},
}


# 2. LOAD DATA
@st.cache_data
def load_data():
    try:
        # 1. Read File
        df = pd.read_csv("dashboard_data.csv", sep=None, engine="python")

        # 2. Clean Column Names
        df.columns = df.columns.str.strip().str.lower()

        # 3. Check for essential columns
        if "iso3" not in df.columns:
            st.error(f"❌ CRITICAL ERROR: 'iso3' column not found.")
            st.stop()

        # 4. Create ID
        df["survey_id"] = df["iso3"] + "_" + df["year"].astype(str)
        return df

    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


df = load_data()

if df is None:
    st.error("❌ 'dashboard_data.csv' not found.")
    st.stop()

# 3. PRE-PROCESSING
r2_cols = list(MODEL_CONFIG.keys())
survey_stats = df.groupby("survey_id")[r2_cols].first().reset_index()

# Find Max R2 and Winner
survey_stats["best_r2"] = survey_stats[r2_cols].max(axis=1)
survey_stats["winning_col"] = survey_stats[r2_cols].idxmax(axis=1)
survey_stats["winning_model"] = survey_stats["winning_col"].map(
    lambda x: MODEL_CONFIG[x]["name"] if pd.notnull(x) else "None"
)

# Sort
survey_stats = survey_stats.sort_values(by="best_r2", ascending=False)
best_score_map = dict(zip(survey_stats["survey_id"], survey_stats["best_r2"]))
winner_map = dict(zip(survey_stats["survey_id"], survey_stats["winning_model"]))
sorted_survey_list = survey_stats["survey_id"].tolist()

# 4. SIDEBAR
st.sidebar.markdown("## Settings")


def format_func(survey_id):
    return f"{survey_id} (R2: {best_score_map[survey_id]:.3f})"


selected_survey = st.sidebar.selectbox(
    "Survey", options=sorted_survey_list, format_func=format_func
)

# 5. MAIN CONTENT
# UPDATED: Added a third tab and renamed the second one
tab1, tab2, tab3 = st.tabs(
    ["📈 Chart Visualization", "🏆 Best Model", "📊 Model Distribution"]
)

# --- TAB 1: CHART VISUALIZATION ---
with tab1:
    dff = df[df["survey_id"] == selected_survey].copy()
    current_scores = {col: dff[col].iloc[0] for col in r2_cols}

    st.markdown(f"### Survey: {selected_survey}")

    # UPDATED: Display R2 scores for EVERY model in a clean table/expander
    with st.expander("Show R² Scores for all models", expanded=True):
        score_data = []
        for r2_col, score in current_scores.items():
            score_data.append(
                {"Model Name": MODEL_CONFIG[r2_col]["name"], "R² Score": score}
            )

        score_df = pd.DataFrame(score_data).sort_values(by="R² Score", ascending=False)

        # Highlight the best score
        st.dataframe(
            score_df.style.background_gradient(subset=["R² Score"], cmap="Greens"),
            use_container_width=True,
            hide_index=True,
        )

    # Prepare Plot Data
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

    # Color Logic
    color_map = {}
    available_colors = px.colors.qualitative.G10
    idx = 0
    for m in plot_df["Model"].unique():
        if m == "TRUE INCOME":
            color_map[m] = "black"
        elif m == "Simulated":
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
    fig.update_traces(
        selector={"name": "TRUE INCOME"},
        mode="lines+markers",
        line=dict(width=3, dash="dot", color="black"),
    )
    if "Simulated" in color_map:
        fig.update_traces(
            selector={"name": "Simulated"}, line=dict(width=3, color="red", dash="dash")
        )

    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: BEST MODEL LIST ---
with tab2:
    st.markdown("### Best Model per Survey")
    st.dataframe(
        survey_stats[
            ["survey_id", "winning_model", "best_r2"]
        ].style.background_gradient(subset=["best_r2"], cmap="Greens"),
        use_container_width=True,
    )

# --- TAB 3: MODEL DISTRIBUTION (NEW) ---
with tab3:
    st.markdown("### Which model wins most often?")

    # Calculate counts and percentages
    win_counts = survey_stats["winning_model"].value_counts().reset_index()
    win_counts.columns = ["Model", "Count"]
    win_counts["Percentage"] = (win_counts["Count"] / win_counts["Count"].sum()) * 100

    # Create Bar Chart
    fig_bar = px.bar(
        win_counts,
        x="Model",
        y="Percentage",
        text_auto=".1f",
        title="Distribution of Best Performing Models",
        color="Model",
        hover_data=["Count"],
    )

    fig_bar.update_layout(yaxis_title="Percentage of Surveys Won (%)", showlegend=False)
    fig_bar.update_traces(texttemplate="%{y:.1f}%", textposition="outside")

    st.plotly_chart(fig_bar, use_container_width=True)
