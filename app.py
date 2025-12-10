import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import copy  # For deep copying dictionaries

# 1. PAGE SETUP
st.set_page_config(layout="wide", page_title="Income Distribution Dashboard")

# --- CONFIGURATION ---
POP_FILENAME = "API_SP.POP.TOTL_DS2_en_csv_v2_246068.csv"
MAIN_FILENAME = "dashboard_data.csv"

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

# --- INIT SESSION STATE FOR LOGGING ---
if "change_log" not in st.session_state:
    st.session_state.change_log = []


# 2. LOAD & MERGE DATA
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(MAIN_FILENAME, sep=None, engine="python")
        df.columns = df.columns.str.strip().str.lower()
        if "iso3" not in df.columns:
            st.error("❌ CRITICAL ERROR: 'iso3' column not found.")
            st.stop()
        df["survey_id"] = df["iso3"] + "_" + df["year"].astype(str)
    except Exception as e:
        st.error(f"Error reading {MAIN_FILENAME}: {e}")
        return None

    if os.path.exists(POP_FILENAME):
        try:
            pop_raw = pd.read_csv(POP_FILENAME, skiprows=4)
            year_cols = [c for c in pop_raw.columns if c.isdigit()]
            keep_cols = ["Country Code"] + year_cols
            pop_subset = pop_raw[keep_cols].copy()
            pop_long = pop_subset.melt(
                id_vars=["Country Code"],
                value_vars=year_cols,
                var_name="year",
                value_name="population",
            )
            pop_long.rename(columns={"Country Code": "iso3"}, inplace=True)
            pop_long["year"] = pd.to_numeric(pop_long["year"], errors="coerce")
            df = pd.merge(df, pop_long, on=["iso3", "year"], how="left")
        except Exception as e:
            st.warning(f"⚠️ Could not process population file: {e}")
            df["population"] = 0
    else:
        st.warning(f"⚠️ Population file not found.")
        df["population"] = 0
    return df


raw_df = load_data()
if raw_df is None:
    st.stop()


# 3. STATS FUNCTIONS
def calculate_r2_manual(y_true, y_pred):
    try:
        y_t = np.array(y_true)
        y_p = np.array(y_pred)
        mask = ~np.isnan(y_t) & ~np.isnan(y_p)
        y_t = y_t[mask]
        y_p = y_p[mask]
        if len(y_t) < 2:
            return -999.0
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        if ss_tot < 1e-9:
            return 0.0
        return 1 - (ss_res / ss_tot)
    except:
        return -999.0


@st.cache_data
def calculate_dynamic_stats(data_subset, log_y=False):
    results = []

    def safe_log(arr):
        return np.log(np.maximum(arr, 0.01))

    for survey_id, group in data_subset.groupby("survey_id"):
        row = {"survey_id": survey_id}
        raw_true = group["true_daily_consumption"].values
        y_true = safe_log(raw_true) if log_y else raw_true

        for model_key, config in MODEL_CONFIG.items():
            if config["pred_col"] in group.columns:
                raw_pred = group[config["pred_col"]].values
                y_pred = safe_log(raw_pred) if log_y else raw_pred
                row[model_key] = calculate_r2_manual(y_true, y_pred)
            else:
                row[model_key] = -999.0
        results.append(row)
    return pd.DataFrame(results)


# 4. SIDEBAR CONTROLS & PARAMS CAPTURE
st.sidebar.markdown("## ⚙️ Settings")

# --- INPUTS ---
scale_mode = st.sidebar.radio(
    "Select Analysis Scale", options=["Linear", "Log-Log", "Log-Linear"], index=0
)
is_linear = scale_mode == "Linear"
is_log_y = scale_mode in ["Log-Log", "Log-Linear"]
is_log_x = scale_mode == "Log-Log"

st.sidebar.markdown("---")
min_pop_mil = st.sidebar.number_input(
    "Min Population (Millions)", 0.0, 1000.0, 0.0, 0.5
)
min_pop_val = min_pop_mil * 1_000_000

st.sidebar.markdown("---")
min_val_default = 1 if is_log_x else 0
min_p, max_p = st.sidebar.slider("Percentile Range", 0, 100, (min_val_default, 100), 1)

if is_log_x and min_p == 0:
    st.sidebar.warning("⚠️ Log-Log scale requires Percentile > 0.")
    min_p = 1

# Thresholds (Moved to sidebar for easier tracking in log)
st.sidebar.markdown("---")
st.sidebar.markdown("**Quality Thresholds**")
c_low, c_high = st.sidebar.columns(2)
with c_low:
    thresh_low = st.number_input("Low <", 0.0, 1.0, 0.70, 0.05)
with c_high:
    thresh_high = st.number_input("High >", 0.0, 1.0, 0.90, 0.05)

# --- FILTERING ---
mask = (raw_df["percentile"] >= min_p) & (raw_df["percentile"] <= max_p)
if min_pop_val > 0:
    mask = mask & (raw_df["population"].fillna(0) >= min_pop_val)
filtered_df = raw_df[mask].copy()

# Excluded list
all_countries = set(raw_df["iso3"].unique())
kept_countries = set(filtered_df["iso3"].unique()) if not filtered_df.empty else set()
excluded_countries = sorted(list(all_countries - kept_countries))

if excluded_countries:
    with st.sidebar.expander(f"❌ Excluded ({len(excluded_countries)})"):
        st.write(", ".join(excluded_countries))

if filtered_df.empty:
    st.error("❌ No data found with current filters.")
    st.stop()

unique_surveys = filtered_df["survey_id"].nunique()
st.sidebar.caption(f"Surveys Included: **{unique_surveys}**")

# --- CALCULATIONS ---
survey_stats = calculate_dynamic_stats(filtered_df, log_y=is_log_y)

r2_cols = list(MODEL_CONFIG.keys())
# Global means
stats_for_avg = survey_stats[r2_cols].replace(-999.0, np.nan)
global_means = stats_for_avg.mean().reset_index()
global_means.columns = ["model_key", "mean_r2"]
global_means["Model"] = global_means["model_key"].map(lambda x: MODEL_CONFIG[x]["name"])
global_means = global_means.sort_values(by="mean_r2", ascending=False)

# Winners
survey_stats["best_r2"] = survey_stats[r2_cols].max(axis=1)
survey_stats["winning_col"] = survey_stats[r2_cols].idxmax(axis=1)
survey_stats["winning_model"] = survey_stats["winning_col"].map(
    lambda x: MODEL_CONFIG[x]["name"] if pd.notnull(x) else "None"
)
survey_stats = survey_stats[survey_stats["best_r2"] > -100].sort_values(
    by="best_r2", ascending=False
)

best_score_map = dict(zip(survey_stats["survey_id"], survey_stats["best_r2"]))
sorted_survey_list = survey_stats["survey_id"].tolist()

if not sorted_survey_list:
    st.warning("Not enough data.")
    st.stop()

# --- PREPARE LOGGING DATA ---
# 1. Define Labels
label_high = f"High (> {thresh_high:.2f})"
label_med = f"Medium ({thresh_low:.2f}-{thresh_high:.2f})"
label_low = f"Low (< {thresh_low:.2f})"


# 2. Categorize Quality
def get_quality_label(r2):
    if r2 >= thresh_high:
        return label_high
    elif r2 >= thresh_low:
        return label_med
    else:
        return label_low


survey_stats["Quality"] = survey_stats["best_r2"].apply(get_quality_label)

# 3. Get Counts
counts = survey_stats["Quality"].value_counts()
count_high = counts.get(label_high, 0)
count_med = counts.get(label_med, 0)
count_low = counts.get(label_low, 0)

# 4. Current State Object
current_params = {
    "scale": scale_mode,
    "min_pop": min_pop_mil,
    "p_range": f"{min_p}-{max_p}",
    "t_low": thresh_low,
    "t_high": thresh_high,
}

current_results = {
    "n_surveys": unique_surveys,
    "high": count_high,
    "med": count_med,
    "low": count_low,
    "label_high": label_high,  # Store labels as they change with thresholds
    "label_med": label_med,
    "label_low": label_low,
}

# --- LOGGING LOGIC ---
# Check if this state is different from the last one logged
should_log = False
if not st.session_state.change_log:
    should_log = True
else:
    last_entry = st.session_state.change_log[-1]
    # Compare inputs
    if current_params != last_entry["params"]:
        should_log = True

if should_log:
    # Calculate Deltas (if previous exists)
    deltas = {}
    changes = []

    if st.session_state.change_log:
        last = st.session_state.change_log[-1]
        last_res = last["results"]
        last_par = last["params"]

        deltas["n_surveys"] = current_results["n_surveys"] - last_res["n_surveys"]
        deltas["high"] = current_results["high"] - last_res["high"]
        deltas["med"] = current_results["med"] - last_res["med"]
        deltas["low"] = current_results["low"] - last_res["low"]

        # Detect what parameter changed
        for k, v in current_params.items():
            if v != last_par[k]:
                changes.append(f"{k}: {last_par[k]} → {v}")
    else:
        # Initial state (deltas are 0)
        deltas = {"n_surveys": 0, "high": 0, "med": 0, "low": 0}
        changes = ["Initial Load"]

    # Append
    st.session_state.change_log.append(
        {
            "id": len(st.session_state.change_log) + 1,
            "params": current_params,
            "results": current_results,
            "deltas": deltas,
            "changes_txt": ", ".join(changes),
        }
    )

# --- PERSISTENT SELECTION ---
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
st.sidebar.markdown(f"### 🌍 Global Average R²")
st.sidebar.dataframe(
    global_means[["Model", "mean_r2"]]
    .style.background_gradient(subset=["mean_r2"], cmap="Greens")
    .format({"mean_r2": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

# 5. MAIN CONTENT
st.title(f"Analysis: {scale_mode} Scale")

tab_options = [
    "📈 Chart Visualization",
    "📊 Model Analysis",
    "🌍 Survey Scatter",
    "📝 Model Equations",
    "📜 Change Log",
]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = tab_options[0]

selected_tab = st.radio(
    "Select View",
    options=tab_options,
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)
st.markdown("---")

# --- VIEW 1: CHART ---
if selected_tab == "📈 Chart Visualization":
    dff = filtered_df[filtered_df["survey_id"] == selected_survey].copy()
    survey_row = survey_stats[survey_stats["survey_id"] == selected_survey].iloc[0]

    current_pop = dff["population"].max()
    pop_str = f"{current_pop/1_000_000:.1f}M" if pd.notnull(current_pop) else "N/A"

    st.markdown(f"### Survey: {selected_survey}")
    st.caption(f"Population: **{pop_str}**")

    with st.expander(f"Show {scale_mode} R² Scores", expanded=True):
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

    if is_log_y:
        plot_df["Prediction"] = np.log(np.maximum(plot_df["Prediction"], 0.01))
        y_axis_label = "Log(Daily Consumption)"
    else:
        y_axis_label = "Daily Consumption ($)"

    if is_log_x:
        plot_df["percentile"] = np.log(np.maximum(plot_df["percentile"], 0.01))
        x_axis_label = "Log(Percentile)"
    else:
        x_axis_label = "Percentile"

    plot_df["Prediction"] = plot_df["Prediction"].round(2)

    color_map = {}
    available_colors = px.colors.qualitative.G10
    idx = 0
    for m in plot_df["Model"].unique():
        if m == "TRUE INCOME":
            color_map[m] = "#FFC107"
        elif m == "0. Simulated":
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
        labels={"percentile": x_axis_label, "Prediction": y_axis_label},
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_traces(
        selector={"name": "TRUE INCOME"},
        mode="markers",
        marker=dict(size=8, color="#FFC107", symbol="circle"),
    )
    if "0. Simulated" in color_map:
        fig.update_traces(
            selector={"name": "0. Simulated"},
            line=dict(width=3, color="red", dash="dash"),
        )
    st.plotly_chart(fig, use_container_width=True)

# --- VIEW 2: MODEL ANALYSIS ---
elif selected_tab == "📊 Model Analysis":
    st.markdown(f"### Winning Models Distribution ({scale_mode})")

    win_counts = survey_stats["winning_model"].value_counts().reset_index()
    win_counts.columns = ["Model", "Count"]
    win_counts["Percentage"] = (win_counts["Count"] / win_counts["Count"].sum()) * 100

    fig_bar = px.bar(
        win_counts, x="Model", y="Percentage", text_auto=".2f", color="Model"
    )
    fig_bar.update_layout(yaxis_title="Percentage Won (%)", showlegend=False)
    fig_bar.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.dataframe(
        survey_stats[["survey_id", "winning_model", "best_r2"]]
        .style.background_gradient(subset=["best_r2"], cmap="Greens")
        .format({"best_r2": "{:.2f}"}),
        use_container_width=True,
    )

# --- VIEW 3: SCATTER OVERVIEW ---
elif selected_tab == "🌍 Survey Scatter":
    st.markdown("### 🌍 Global Performance Overview")

    # Merge Meta
    survey_meta = filtered_df[
        ["survey_id", "iso3", "year", "gdp_pcap", "gini_disp", "population"]
    ].drop_duplicates()
    scatter_df = pd.merge(survey_stats, survey_meta, on="survey_id", how="inner")
    scatter_df["pop_m"] = scatter_df["population"] / 1_000_000

    # Colors
    color_map_scatter = {
        label_high: "#28a745",
        label_med: "#ffc107",
        label_low: "#dc3545",
    }

    col_t1, col_t2 = st.columns([1, 4])
    with col_t1:
        use_pop_size = st.checkbox("Size by Pop?", value=False)

    size_col = "pop_m" if use_pop_size else None
    size_max_val = 15 if use_pop_size else None

    fig_scatter = px.scatter(
        scatter_df,
        x="gdp_pcap",
        y="gini_disp",
        color="Quality",
        color_discrete_map=color_map_scatter,
        hover_name="survey_id",
        size=size_col,
        size_max=size_max_val,
        hover_data={
            "best_r2": ":.3f",
            "winning_model": True,
            "Quality": False,
            "gdp_pcap": ":,.0f",
        },
        height=500,
        title=f"Model Accuracy vs. Economic Indicators",
    )
    fig_scatter.update_layout(
        template="plotly_white", legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Fit Quality Distribution")

    quality_counts = scatter_df["Quality"].value_counts().reset_index()
    quality_counts.columns = ["Quality", "Count"]
    quality_counts["Percentage"] = (
        quality_counts["Count"] / quality_counts["Count"].sum()
    ) * 100

    cat_order = [label_high, label_med, label_low]

    fig_bar_qual = px.bar(
        quality_counts,
        x="Quality",
        y="Count",
        color="Quality",
        text_auto=True,
        color_discrete_map=color_map_scatter,
        category_orders={"Quality": cat_order},
        hover_data={"Percentage": ":.1f"},
        height=400,
    )
    fig_bar_qual.update_layout(
        yaxis_title="Number of Surveys", template="plotly_white", showlegend=False
    )
    fig_bar_qual.update_traces(texttemplate="%{y} <br>(%{customdata[0]:.1f}%)")
    st.plotly_chart(fig_bar_qual, use_container_width=True)

# --- VIEW 4: MODEL EQUATIONS ---
elif selected_tab == "📝 Model Equations":
    st.markdown("### 📝 Model Specifications")

    if scale_mode == "Linear":
        st.caption("ℹ️ **Mode: Linear.**")
        Y_term = r"Y_{icy}"
        P_term = r"P_i"
    elif scale_mode == "Log-Log":
        st.caption("⚠️ **Mode: Log-Log.**")
        Y_term = r"\ln(Y_{icy})"
        P_term = r"\ln(P_i)"
    elif scale_mode == "Log-Linear":
        st.caption("⚠️ **Mode: Log-Linear.**")
        Y_term = r"\ln(Y_{icy})"
        P_term = r"P_i"

    st.markdown(rf"**Notation:** ${Y_term}$ Consumption, ${P_term}$ Percentile.")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.info("##### 1. Global Base")
        st.markdown(
            rf"$$ {Y_term} = \beta_0 + \beta_1 \text{{GDP}} + \beta_2 {P_term} $$"
        )
        st.info("##### 2. Gini Interaction")
        st.markdown(
            rf"$$ {Y_term} = \dots + \beta_3 {P_term} + \beta_4 ({P_term} \times \text{{Gini}}) $$"
        )
    with c2:
        st.info("##### 3. Country FE")
        st.markdown(rf"$$ {Y_term} = \alpha_c + \gamma_c {P_term} + \dots $$")
        st.warning("##### 0. Simulated")
        st.markdown("Baseline synthetic data.")

    st.divider()
    st.latex(rf"{Y_term} = \alpha_{{cy}} + \sum \beta ({P_term})^k")
    cols = st.columns(4)
    cols[0].latex(rf"{Y_term} \sim {P_term}")
    cols[1].latex(rf"{Y_term} \sim {P_term}^2")
    cols[2].latex(rf"{Y_term} \sim {P_term}^3")
    cols[3].latex(rf"{Y_term} \sim {P_term}^4")

# --- VIEW 5: CHANGE LOG (NEW) ---
elif selected_tab == "📜 Change Log":
    st.markdown("### 📜 Session History")
    st.caption(
        "Tracks how your global statistics change as you modify settings (Population, Thresholds, Scale, etc)."
    )

    if not st.session_state.change_log:
        st.info("No changes recorded yet.")
    else:
        # Show newest first
        for i, entry in enumerate(reversed(st.session_state.change_log)):
            res = entry["results"]
            delta = entry["deltas"]
            params = entry["params"]

            # Helper to format delta
            def fmt_delta(val):
                if val > 0:
                    return f"(+{val})"
                elif val < 0:
                    return f"({val})"
                else:
                    return ""  # Don't show if 0

            # Card-like container
            with st.container():
                st.markdown(f"#### Step {entry['id']}: {entry['changes_txt']}")

                c1, c2, c3, c4 = st.columns(4)

                # Metric 1: Survey Count
                c1.metric("Surveys", res["n_surveys"], fmt_delta(delta["n_surveys"]))

                # Metric 2: High Quality
                # Note: We use the stored label from that snapshot in case thresholds changed
                c2.metric(
                    f"🟢 {res['label_high']}", res["high"], fmt_delta(delta["high"])
                )

                # Metric 3: Med Quality
                c3.metric(f"🟡 {res['label_med']}", res["med"], fmt_delta(delta["med"]))

                # Metric 4: Low Quality
                c4.metric(
                    f"🔴 {res['label_low']}",
                    res["low"],
                    fmt_delta(
                        delta["low"],
                    ),
                )

                st.divider()

        if st.button("Clear Log"):
            st.session_state.change_log = []
            st.rerun()
