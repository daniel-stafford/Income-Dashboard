import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import geopandas as gpd

# 1. PAGE SETUP
st.set_page_config(layout="wide", page_title="Income Distribution Dashboard")

# --- CONFIGURATION & PATHS ---
# Define the data folder relative to app.py
DATA_FOLDER = "data"

# Join paths securely so they work on Mac (local) and Linux (Streamlit Cloud)
POP_FILENAME = os.path.join(DATA_FOLDER, "API_SP.POP.TOTL_DS2_en_csv_v2_246068.csv")
MAIN_FILENAME = os.path.join(DATA_FOLDER, "dashboard_data.csv")
PATH_LMICS = os.path.join(DATA_FOLDER, "CLASS_2025_10_07.xlsx")
PATH_MAPPING = os.path.join(DATA_FOLDER, "WB_country_iso3_mapping.xls")

# Shapefile Path (pointing to the folder moved inside 'data')
PATH_SHAPEFILE = os.path.join(
    DATA_FOLDER, "ne_110m_admin_0_countries", "ne_110m_admin_0_countries.shp"
)

# --- NAMING CONVENTION ---
MODEL_CONFIG = {
    "r2_simulated": {"pred_col": "simulated_daily_consumption", "name": "0. Simulated"},
    "r2_m1_global": {"pred_col": "pred_m1_global", "name": "1. Global Fixed Slope"},
    "r2_m2_gini": {"pred_col": "pred_m2_gini", "name": "2. Gini Interaction"},
    "r2_m3_cntry": {"pred_col": "pred_m3_cntry", "name": "3. Country-Specific Trends"},
    "r2_m4_linear": {"pred_col": "pred_m4_linear", "name": "4. Survey Fit (Linear)"},
    "r2_m5_quad": {
        "pred_col": "pred_m5_quadratic",
        "name": "5. Survey Fit (Quadratic)",
    },
    "r2_m6_cubic": {"pred_col": "pred_m6_cubic", "name": "6. Survey Fit (Cubic)"},
    "r2_m7_quartic": {"pred_col": "pred_m7_quartic", "name": "7. Survey Fit (Quartic)"},
}

COLOR_PALETTE = {
    "0. Simulated": "#E31A1C",
    "1. Global Fixed Slope": "#1F78B4",
    "2. Gini Interaction": "#A6CEE3",
    "3. Country-Specific Trends": "#33A02C",
    "4. Survey Fit (Linear)": "#B2DF8A",
    "5. Survey Fit (Quadratic)": "#6A3D9A",
    "6. Survey Fit (Cubic)": "#FDBF6F",
    "7. Survey Fit (Quartic)": "#FF7F00",
    "TRUE INCOME": "#FFD700",
    "None": "#999999",
}

MODEL_LOGICAL_ORDER = [config["name"] for config in MODEL_CONFIG.values()] + [
    "TRUE INCOME",
    "None",
]

if "change_log" not in st.session_state:
    st.session_state.change_log = []

# 2. LOAD DATA FUNCTIONS


@st.cache_data
def load_main_data():
    try:
        df = pd.read_csv(MAIN_FILENAME, sep=None, engine="python")
        df.columns = df.columns.str.strip().str.lower()
        if "iso3" not in df.columns:
            st.error("❌ CRITICAL ERROR: 'iso3' column not found.")
            st.stop()
        df["survey_id"] = df["iso3"] + "_" + df["year"].astype(str)
    except Exception as e:
        st.error(f"Error reading {MAIN_FILENAME}: {e}")
        return None, None

    # Load Pop for Merging
    if os.path.exists(POP_FILENAME):
        try:
            pop_raw = pd.read_csv(POP_FILENAME, skiprows=4)
            year_cols = [c for c in pop_raw.columns if c.isdigit()]
            keep_cols = ["Country Code"] + year_cols
            pop_subset = pop_raw[keep_cols].copy()

            # For Main DF (Long Format)
            pop_long = pop_subset.melt(
                id_vars=["Country Code"],
                value_vars=year_cols,
                var_name="year",
                value_name="population",
            )
            pop_long.rename(columns={"Country Code": "iso3"}, inplace=True)
            pop_long["year"] = pd.to_numeric(pop_long["year"], errors="coerce")
            df = pd.merge(df, pop_long, on=["iso3", "year"], how="left")

            return df, pop_raw  # Return raw pop for map usage
        except Exception as e:
            st.warning(f"⚠️ Could not process population file: {e}")
            df["population"] = 0
            return df, None
    else:
        st.warning(f"⚠️ Population file not found.")
        df["population"] = 0
        return df, None


# Load Map Geospatial Data
@st.cache_data
def load_map_resources():
    try:
        # 1. LMIC Data
        df_lmic = pd.read_excel(PATH_LMICS)
        df_lmic = df_lmic[df_lmic["Income group"] != "High income"]
        lmic_isos = set(df_lmic["Code"].astype(str).str.strip().str.upper())

        # 2. Shapefile
        # Check if shapefile actually exists at the relative path before loading
        if not os.path.exists(PATH_SHAPEFILE):
            return None, None, f"Shapefile not found at: {PATH_SHAPEFILE}"

        gdf_admin0 = gpd.read_file(PATH_SHAPEFILE)

        # Convert to EPSG:4326 (Lat/Lon)
        if gdf_admin0.crs != "EPSG:4326":
            gdf_admin0 = gdf_admin0.to_crs(epsg=4326)

        # Simplify geometry significantly for web rendering speed
        gdf_admin0["geometry"] = gdf_admin0.simplify(0.05, preserve_topology=True)
        gdf_admin0 = gdf_admin0.rename(columns={"ISO_A3": "iso3"})

        # 3. Country Names Mapping
        df_names = pd.read_excel(PATH_MAPPING)
        if "country" in df_names.columns and "iso3" in df_names.columns:
            df_names = df_names[["iso3", "country"]].rename(
                columns={"country": "country_name"}
            )
        else:
            # Fallback
            df_names.columns = [c.lower() for c in df_names.columns]
            df_names = df_names[["iso3", "country"]].rename(
                columns={"country": "country_name"}
            )

        return lmic_isos, gdf_admin0, df_names
    except Exception as e:
        return None, None, str(e)


raw_df, raw_pop_df = load_main_data()
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


# 4. SIDEBAR CONTROLS
st.sidebar.markdown("## ⚙️ Settings")

scale_mode = st.sidebar.radio(
    "Select Analysis Scale", options=["Linear", "Log-Log", "Log-Linear"], index=2
)
is_linear = scale_mode == "Linear"
is_log_y = scale_mode in ["Log-Log", "Log-Linear"]
is_log_x = scale_mode == "Log-Log"

st.sidebar.markdown("---")
# 1. Population Input
min_pop_mil = st.sidebar.number_input(
    "Min Population (Millions)", 0.0, 1000.0, 1.0, 0.5
)
min_pop_val = min_pop_mil * 1_000_000

# 2. Percentile Input
default_range = (1, 80)
min_p, max_p = st.sidebar.slider("Percentile Range", 0, 100, default_range, 1)

if is_log_x and min_p == 0:
    min_p = 1

# --- FILTERING LOGIC ---
mask = (raw_df["percentile"] >= min_p) & (raw_df["percentile"] <= max_p)
if min_pop_val > 0:
    mask = mask & (raw_df["population"].fillna(0) >= min_pop_val)
filtered_df = raw_df[mask].copy()

unique_surveys = filtered_df["survey_id"].nunique()

# --- EXCLUDED SURVEYS LIST ---
all_survey_ids = set(raw_df["survey_id"].unique())
kept_survey_ids = set(filtered_df["survey_id"].unique())
excluded_ids = sorted(list(all_survey_ids - kept_survey_ids))

if excluded_ids:
    with st.sidebar.expander(f"❌ Excluded Surveys ({len(excluded_ids)})"):
        st.caption(f"Filtered out (Pop < {min_pop_mil}M or Empty Range)")
        st.write(", ".join(excluded_ids))

st.sidebar.markdown("---")
st.sidebar.markdown("**$R^2$ Quality Thresholds**")
c_low, c_high = st.sidebar.columns(2)
with c_low:
    thresh_low = st.number_input("Low <", 0.0, 1.0, 0.70, 0.05)
with c_high:
    thresh_high = st.number_input("High >", 0.0, 1.0, 0.90, 0.05)

# --- CALCULATIONS ---
survey_stats = calculate_dynamic_stats(filtered_df, log_y=is_log_y)
r2_cols = list(MODEL_CONFIG.keys())
stats_for_avg = survey_stats[r2_cols].replace(-999.0, np.nan)
global_means = stats_for_avg.mean().reset_index()
global_means.columns = ["model_key", "mean_r2"]
global_means["Model"] = global_means["model_key"].map(lambda x: MODEL_CONFIG[x]["name"])
global_means = global_means.sort_values(by="mean_r2", ascending=False)

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
    "label_high": label_high,
    "label_med": label_med,
    "label_low": label_low,
}

# --- LOGGING LOGIC ---
should_log = False
if not st.session_state.change_log:
    should_log = True
else:
    last_entry = st.session_state.change_log[-1]
    if current_params != last_entry["params"]:
        should_log = True

if should_log:
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

        for k, v in current_params.items():
            if v != last_par[k]:
                changes.append(f"{k}: {last_par[k]} → {v}")
    else:
        deltas = {"n_surveys": 0, "high": 0, "med": 0, "low": 0}
        changes = ["Initial Load"]

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
    st.session_state.last_selected_survey = (
        sorted_survey_list[0] if sorted_survey_list else None
    )


def update_survey_selection():
    st.session_state.last_selected_survey = st.session_state.survey_widget


try:
    pre_selected_index = sorted_survey_list.index(st.session_state.last_selected_survey)
except:
    pre_selected_index = 0


def format_func(survey_id):
    score = best_score_map.get(survey_id, 0)
    return f"{survey_id} (R²: {score:.2f})"


st.sidebar.markdown("---")
st.sidebar.markdown("**Select Survey**")
if sorted_survey_list:
    selected_survey = st.sidebar.selectbox(
        "Select Survey",
        options=sorted_survey_list,
        format_func=format_func,
        index=pre_selected_index,
        key="survey_widget",
        on_change=update_survey_selection,
        label_visibility="collapsed",
    )
else:
    st.sidebar.error("No surveys available.")

st.sidebar.markdown("---")
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
    "🗺️ Coverage Map",
    "📝 Model Equations",
    "📜 Change Log",
]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🗺️ Coverage Map"

selected_tab = st.radio(
    "Select View",
    options=tab_options,
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)
st.markdown("---")


# ==========================================
# MAP LOGIC (Shapefile Version)
# ==========================================
def get_map_dataframe(
    pop_threshold, lmic_isos, gdf_admin0, df_names, raw_pop_df, cons_df
):
    # 1. Get Consumption ISOs
    cons_isos = set(cons_df["iso3"].astype(str).str.strip().str.upper())

    # 2. Prepare Pop Data
    year_cols = [c for c in raw_pop_df.columns if c.isdigit()]
    latest_year = year_cols[-1]
    df_pop = raw_pop_df[["Country Code", latest_year]].copy()
    df_pop.columns = ["iso3", "population"]

    # 3. Create DataFrame
    # Use the shapefile (gdf_admin0) as the base
    gdf_map = gdf_admin0.copy()
    if "population" in gdf_map.columns:
        gdf_map = gdf_map.drop(columns=["population"])
    if "country_name" in gdf_map.columns:
        gdf_map = gdf_map.drop(columns=["country_name"])

    gdf_map = gdf_map.merge(df_pop, on="iso3", how="left")
    gdf_map = gdf_map.merge(df_names, on="iso3", how="left")
    gdf_map["country_name"] = gdf_map["country_name"].fillna(gdf_map["iso3"])

    # 4. Categorization Logic
    def get_status(row):
        iso = str(row["iso3"]).strip().upper() if row["iso3"] else ""
        in_lmic = iso in lmic_isos
        in_cons = iso in cons_isos
        pop = row["population"] if pd.notnull(row["population"]) else 0
        is_small = pop < pop_threshold

        if not in_lmic and not in_cons:
            return "Excluded (High Income)"

        # Priority Logic
        if in_lmic:
            if in_cons:
                if is_small:
                    return "Excluded (Small Pop)"  # Small but has data
                return "Covered (Data and LMIC)"  # Standard coverage
            else:
                return "Missing Data (LMIC)"  # LMIC but no data

        # Not LMIC but in Consumption (Rare)
        return "Non-LMIC with Data"

    gdf_map["Status"] = gdf_map.apply(get_status, axis=1)

    # Calculate Stats
    lmic_mask = gdf_map["iso3"].apply(lambda x: str(x).upper() in lmic_isos)
    total_lmic_pop = gdf_map.loc[lmic_mask, "population"].sum()

    valid_mask = gdf_map["Status"] == "Covered (Data and LMIC)"
    covered_valid_pop = gdf_map.loc[valid_mask, "population"].sum()

    pct = (covered_valid_pop / total_lmic_pop * 100) if total_lmic_pop > 0 else 0

    return gdf_map, pct, covered_valid_pop, total_lmic_pop


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

    fig = px.line(
        plot_df,
        x="percentile",
        y="Prediction",
        color="Model",
        color_discrete_map=COLOR_PALETTE,
        category_orders={"Model": MODEL_LOGICAL_ORDER},
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
        marker=dict(size=8, symbol="circle"),
    )

    sim_name = MODEL_CONFIG["r2_simulated"]["name"]
    fig.update_traces(
        selector={"name": sim_name},
        line=dict(width=3, dash="dash"),
    )

    st.plotly_chart(fig, use_container_width=True)

# --- VIEW 2: MODEL ANALYSIS ---
elif selected_tab == "📊 Model Analysis":
    st.markdown(f"### Winning Models Distribution ({scale_mode})")
    win_counts = survey_stats["winning_model"].value_counts().reset_index()
    win_counts.columns = ["Model", "Count"]
    win_counts["Percentage"] = (win_counts["Count"] / win_counts["Count"].sum()) * 100
    win_counts = win_counts.sort_values(by="Percentage", ascending=False)
    sorted_models_by_count = win_counts["Model"].tolist()

    fig_bar = px.bar(
        win_counts,
        x="Model",
        y="Percentage",
        text_auto=".2f",
        color="Model",
        color_discrete_map=COLOR_PALETTE,
        category_orders={"Model": sorted_models_by_count},
    )
    fig_bar.update_layout(yaxis_title="Percentage Won (%)", showlegend=True)
    fig_bar.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Survey Performance")

    sort_order = st.radio(
        "Sort Order:",
        options=["Best to Worst (High -> Low)", "Worst to Best (Low -> High)"],
        horizontal=True,
    )

    if "Best to Worst" in sort_order:
        chart_df = survey_stats.sort_values(by="best_r2", ascending=False)
    else:
        chart_df = survey_stats.sort_values(by="best_r2", ascending=True)

    chart_df = chart_df.reset_index(drop=True)
    chart_df["rank_index"] = chart_df.index
    chart_df["visual_r2"] = np.maximum(chart_df["best_r2"], -0.1)

    total_width = max(1000, len(chart_df) * 15)

    fig_scroll = px.bar(
        chart_df,
        x="rank_index",
        y="visual_r2",
        color="winning_model",
        color_discrete_map=COLOR_PALETTE,
        category_orders={"winning_model": MODEL_LOGICAL_ORDER},
        hover_name="survey_id",
        hover_data={"best_r2": ":.3f", "visual_r2": False, "rank_index": False},
        width=total_width,
        height=600,
        text_auto=".3f",
        title=f"All Surveys ({len(chart_df)})",
    )

    fig_scroll.update_layout(
        xaxis_title="Survey Count (Index)",
        yaxis_title="Best R² Score (Capped at -0.1)",
        legend_title="Winning Model",
        margin=dict(l=50, r=50, t=50, b=50),
    )
    fig_scroll.update_traces(textposition="outside")
    st.plotly_chart(fig_scroll, use_container_width=False)

# --- VIEW 3: SCATTER OVERVIEW ---
elif selected_tab == "🌍 Survey Scatter":
    st.markdown("### 🌍 Global Performance Overview")
    survey_meta = filtered_df[
        ["survey_id", "iso3", "year", "gdp_pcap", "gini_disp", "population"]
    ].drop_duplicates()
    scatter_df = pd.merge(survey_stats, survey_meta, on="survey_id", how="inner")
    scatter_df["pop_m"] = scatter_df["population"] / 1_000_000

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
        hover_data={"best_r2": ":.3f", "winning_model": True, "Quality": False},
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

# --- VIEW 4: MAP (SHAPEFILE VERSION) ---
elif selected_tab == "🗺️ Coverage Map":
    st.markdown("### 🗺️ Data Coverage Map")
    st.caption(
        f"Small countries (< {min_pop_mil}M) are shown in **Grey/Yellow** to distinguish them from full coverage."
    )

    lmic_isos, gdf_admin0, df_names = load_map_resources()

    if lmic_isos is None:
        st.error(f"❌ Failed to load map resources. Error: {df_names}")
    else:
        with st.spinner("Generating Map..."):
            gdf_map, pct, pop_cov, pop_tot = get_map_dataframe(
                min_pop_val, lmic_isos, gdf_admin0, df_names, raw_pop_df, raw_df
            )

            # KPI Cards
            k1, k2, k3 = st.columns(3)
            k1.metric("Coverage (Pop)", f"{pct:.1f}%")
            k2.metric("Covered People", f"{pop_cov/1e9:.2f}B")
            k3.metric("Total LMIC Pop", f"{pop_tot/1e9:.2f}B")

            # Map Colors
            color_map = {
                "Covered (Data and LMIC)": "#2ca02c",  # Green
                "Excluded (Small Pop)": "#ffd700",  # Gold (Distinction for Small)
                "Missing Data (LMIC)": "#d62728",  # Red
                "Non-LMIC with Data": "#1f77b4",  # Blue
                "Excluded (High Income)": "#eeeeee",  # Light Grey
            }

            fig_map = px.choropleth(
                gdf_map,
                geojson=gdf_map.geometry,
                locations=gdf_map.index,
                color="Status",
                color_discrete_map=color_map,
                hover_name="country_name",
                hover_data={"population": ":,.0f", "iso3": True},
                projection="natural earth",
                title="Global Coverage Status",
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(
                height=600,
                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_map, use_container_width=True)

# --- VIEW 5: MODEL EQUATIONS ---
elif selected_tab == "📝 Model Equations":
    st.markdown("### 📝 Model Specifications")
    st.write(
        "Below are the equations and R code (`fixest` package) for all models used in the analysis."
    )

    # Notation Block
    st.info(
        "ℹ️ **Notation:** $Y_{p,c,y}$ = Consumption, $P$ = Percentile Rank. Subscripts: $p$ (percentile), $c$ (country), $y$ (year)."
    )

    if scale_mode == "Linear":
        Y_term = "Y_{p,c,y}"
        P_term = "P"
    elif scale_mode == "Log-Log":
        Y_term = r"\ln(Y_{p,c,y})"
        P_term = r"\ln(P)"
    elif scale_mode == "Log-Linear":
        Y_term = r"\ln(Y_{p,c,y})"
        P_term = "P"

    # --- MODEL 0: SIMULATED ---
    st.markdown("---")
    st.subheader("0. Simulated")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(r"**Log-Normal Simulation:**")
        st.latex(r"Y \sim \text{LogNormal}(\mu_{c,y}, \sigma_{c,y})")
    with c2:
        st.markdown("**Description:**")
        st.write(
            "Consumption is simulated using a Log-Normal distribution derived from summary statistics (Mean and Gini) rather than regression."
        )

    # --- MODEL 1: GLOBAL FIXED SLOPE ---
    st.markdown("---")
    st.subheader("1. Global Fixed Slope")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(
            Y_term
            + r" = \beta_0 + \beta_1 \text{GDP}_{c,y} + \beta_2 "
            + P_term
            + r" + \epsilon"
        )
    with c2:
        st.code("feols(Y ~ gdp + pct, cluster = ~iso3)", language="r")

    # --- MODEL 2: GINI INTERACTION ---
    st.markdown("---")
    st.subheader("2. Gini Interaction")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(
            Y_term
            + r" = \beta_0 + \beta_1 \text{GDP} + \beta_2 \text{Gini} + \beta_3 ("
            + P_term
            + r" \times \text{Gini}) + \epsilon"
        )
    with c2:
        st.code("feols(Y ~ gdp + gini + pct * gini, cluster=~iso3)", language="r")

    # --- MODEL 3: COUNTRY SPECIFIC ---
    st.markdown("---")
    st.subheader("3. Country-Specific Trends")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(
            Y_term
            + r" = \alpha_c + \beta_1 \text{GDP} + \beta_{2,c} "
            + P_term
            + r" + \epsilon"
        )
    with c2:
        st.code("feols(Y ~ gdp | iso3[pct])", language="r")

    # --- MODEL 4: LINEAR ---
    st.markdown("---")
    st.subheader("4. Survey Fit (Linear)")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(Y_term + r" = \beta_0 + \beta_1 " + P_term + r" + \epsilon")
    with c2:
        st.code("lm(Y ~ pct)", language="r")

    # --- MODEL 5: QUADRATIC ---
    st.markdown("---")
    st.subheader("5. Survey Fit (Quadratic)")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(
            Y_term
            + r" = \beta_0 + \beta_1 "
            + P_term
            + r" + \beta_2 "
            + P_term
            + r"^2 + \epsilon"
        )
    with c2:
        st.code("lm(Y ~ poly(pct, 2))", language="r")

    # --- MODEL 6: CUBIC ---
    st.markdown("---")
    st.subheader("6. Survey Fit (Cubic)")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(
            Y_term
            + r" = \beta_0 + \beta_1 "
            + P_term
            + r" + \beta_2 "
            + P_term
            + r"^2 + \beta_3 "
            + P_term
            + r"^3 + \epsilon"
        )
    with c2:
        st.code("lm(Y ~ poly(pct, 3))", language="r")

    # --- MODEL 7: QUARTIC ---
    st.markdown("---")
    st.subheader("7. Survey Fit (Quartic)")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(
            Y_term
            + r" = \beta_0 + \sum_{k=1}^{4} \beta_k ("
            + P_term
            + r")^k + \epsilon"
        )
    with c2:
        st.code("lm(Y ~ poly(pct, 4))", language="r")


# --- VIEW 6: CHANGE LOG ---
elif selected_tab == "📜 Change Log":
    st.markdown("### 📜 Session History")
    if not st.session_state.change_log:
        st.info("No changes recorded yet.")
    else:
        for i, entry in enumerate(reversed(st.session_state.change_log)):
            res = entry["results"]
            delta = entry["deltas"]
            with st.container():
                st.markdown(f"#### Step {entry['id']}: {entry['changes_txt']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Surveys",
                    res["n_surveys"],
                    delta=delta["n_surveys"] if delta["n_surveys"] != 0 else None,
                )
                c2.metric(
                    f"🟢 {res['label_high']}",
                    res["high"],
                    delta=delta["high"] if delta["high"] != 0 else None,
                )
                c3.metric(
                    f"🟡 {res['label_med']}",
                    res["med"],
                    delta=delta["med"] if delta["med"] != 0 else None,
                )
                c4.metric(
                    f"🔴 {res['label_low']}",
                    res["low"],
                    delta=delta["low"] if delta["low"] != 0 else None,
                )
                st.divider()
        if st.button("Clear Log"):
            st.session_state.change_log = []
            st.rerun()
