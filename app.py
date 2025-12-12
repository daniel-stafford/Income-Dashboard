import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import geopandas as gpd

# 1. PAGE SETUP
st.set_page_config(layout="wide", page_title="Income Distribution Dashboard")

# --- INJECT CUSTOM CSS FOR SIDEBAR WIDTH ---
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 400px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CONFIGURATION & PATHS ---
DATA_FOLDER = "data"

POP_FILENAME = os.path.join(DATA_FOLDER, "API_SP.POP.TOTL_DS2_en_csv_v2_246068.csv")
MAIN_FILENAME = os.path.join(DATA_FOLDER, "dashboard_data.csv")
PATH_LMICS = os.path.join(DATA_FOLDER, "CLASS_2025_10_07.xlsx")
PATH_MAPPING = os.path.join(DATA_FOLDER, "WB_country_iso3_mapping.xls")
PATH_SHAPEFILE = os.path.join(
    DATA_FOLDER, "ne_110m_admin_0_countries", "ne_110m_admin_0_countries.shp"
)

# --- FULL MODEL DEFINITION ---
FULL_MODEL_CONFIG = {
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
    "r2_m8_segmented": {
        "pred_col": "pred_m8_segmented",
        "name": "8. Segmented (Piecewise)",
    },
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
    "8. Segmented (Piecewise)": "#E7298A",
    "TRUE INCOME": "#FFD700",
    "None": "#999999",
}

FULL_MODEL_LOGICAL_ORDER = [config["name"] for config in FULL_MODEL_CONFIG.values()] + [
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
            return df, pop_raw
        except Exception as e:
            st.warning(f"⚠️ Could not process population file: {e}")
            df["population"] = 0
            return df, None
    else:
        st.warning(f"⚠️ Population file not found.")
        df["population"] = 0
        return df, None


@st.cache_data
def load_map_resources():
    try:
        df_lmic = pd.read_excel(PATH_LMICS)
        df_lmic = df_lmic[df_lmic["Income group"] != "High income"]
        lmic_isos = set(df_lmic["Code"].astype(str).str.strip().str.upper())

        if not os.path.exists(PATH_SHAPEFILE):
            return None, None, f"Shapefile not found at: {PATH_SHAPEFILE}"

        gdf_admin0 = gpd.read_file(PATH_SHAPEFILE)
        if gdf_admin0.crs != "EPSG:4326":
            gdf_admin0 = gdf_admin0.to_crs(epsg=4326)

        gdf_admin0["geometry"] = gdf_admin0.simplify(0.05, preserve_topology=True)
        gdf_admin0 = gdf_admin0.rename(columns={"ISO_A3": "iso3"})

        df_names = pd.read_excel(PATH_MAPPING)
        if "country" in df_names.columns:
            df_names = df_names.rename(columns={"country": "country_name"})
        else:
            df_names.columns = [c.lower() for c in df_names.columns]
            df_names = df_names.rename(columns={"country": "country_name"})

        return lmic_isos, gdf_admin0, df_names[["iso3", "country_name"]]
    except Exception as e:
        return None, None, str(e)


raw_df, raw_pop_df = load_main_data()
if raw_df is None:
    st.stop()


# 3. STATS FUNCTIONS (LOG CALCULATION)
def calculate_log_metrics(y_true, y_pred):
    """
    Calculates R2 and RMSE on the LOG scale.
    This effectively calculates Log-R2 and Log-RMSE on the fly.
    """
    try:
        # 1. Log Transform (with safety floor for tiny/zero values)
        # Using 1e-5 floor ensures we don't get -inf, though predictions should be positive.
        y_t_log = np.log(np.maximum(y_true, 1e-5))
        y_p_log = np.log(np.maximum(y_pred, 1e-5))

        # 2. Mask NaNs
        mask = ~np.isnan(y_t_log) & ~np.isnan(y_p_log)
        y_t_log, y_p_log = y_t_log[mask], y_p_log[mask]

        if len(y_t_log) < 2:
            return -999.0, 999.0

        # 3. R2 Calculation
        ss_res = np.sum((y_t_log - y_p_log) ** 2)
        ss_tot = np.sum((y_t_log - np.mean(y_t_log)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot >= 1e-9 else 0.0

        # 4. RMSE Calculation
        rmse = np.sqrt(np.mean((y_t_log - y_p_log) ** 2))

        return r2, rmse
    except:
        return -999.0, 999.0


@st.cache_data
def calculate_dynamic_stats(data_subset, active_config):
    """
    Iterates through surveys and calculates metrics using the log-transformed data.
    """
    results = []

    for survey_id, group in data_subset.groupby("survey_id"):
        row = {"survey_id": survey_id}

        # Raw Dollar Values
        y_true_dollars = group["true_daily_consumption"].values

        # Only iterate through ACTIVE models
        for model_key, config in active_config.items():
            if config["pred_col"] in group.columns:
                # Raw Dollar Predictions
                y_pred_dollars = group[config["pred_col"]].values

                # Calculate metrics on LOGS
                r2_log, rmse_log = calculate_log_metrics(y_true_dollars, y_pred_dollars)

                row[model_key] = r2_log
                row[f"{model_key}_rmse"] = rmse_log
            else:
                row[model_key] = -999.0
                row[f"{model_key}_rmse"] = 999.0
        results.append(row)
    return pd.DataFrame(results)


# 4. SIDEBAR CONTROLS
st.sidebar.markdown("## ⚙️ Settings")

# --- MODEL SELECTION FILTER ---
st.sidebar.markdown("**Select Models to Include:**")
available_model_names = [v["name"] for v in FULL_MODEL_CONFIG.values()]
selected_model_names = st.sidebar.multiselect(
    "Active Models",
    options=available_model_names,
    default=available_model_names,  # All selected by default
    label_visibility="collapsed",
)

# Create the ACTIVE configuration based on selection
ACTIVE_MODEL_CONFIG = {
    k: v for k, v in FULL_MODEL_CONFIG.items() if v["name"] in selected_model_names
}
MODEL_LOGICAL_ORDER = [config["name"] for config in ACTIVE_MODEL_CONFIG.values()] + [
    "TRUE INCOME",
    "None",
]
if not ACTIVE_MODEL_CONFIG:
    st.error("Please select at least one model in the sidebar.")
    st.stop()
# -----------------------------------

scale_mode = st.sidebar.radio(
    "Select Analysis Scale", options=["Linear", "Log-Log", "Log-Linear"], index=2
)
is_linear = scale_mode == "Linear"
is_log_y = scale_mode in ["Log-Log", "Log-Linear"]
is_log_x = scale_mode == "Log-Log"

st.sidebar.markdown("---")
min_pop_mil = st.sidebar.number_input(
    "Min Population (Millions)", 0.0, 1000.0, 1.0, 0.5
)
min_pop_val = min_pop_mil * 1_000_000
min_p, max_p = st.sidebar.slider("Percentile Range", 0, 100, (1, 100), 1)
if is_log_x and min_p == 0:
    min_p = 1

# Filtering
mask = (raw_df["percentile"] >= min_p) & (raw_df["percentile"] <= max_p)
if min_pop_val > 0:
    mask = mask & (raw_df["population"].fillna(0) >= min_pop_val)
filtered_df = raw_df[mask].copy()
unique_surveys = filtered_df["survey_id"].nunique()

all_survey_ids = set(raw_df["survey_id"].unique())
kept_survey_ids = set(filtered_df["survey_id"].unique())
excluded_ids = sorted(list(all_survey_ids - kept_survey_ids))
if excluded_ids:
    with st.sidebar.expander(f"❌ Excluded Surveys ({len(excluded_ids)})"):
        st.write(", ".join(excluded_ids))

st.sidebar.markdown("---")
st.sidebar.markdown("**$R^2$ Quality Thresholds**")
c_low, c_high = st.sidebar.columns(2)
thresh_low = c_low.number_input("Medium (≥)", 0.0, 1.0, 0.70, 0.05)
thresh_high = c_high.number_input("High (≥)", 0.0, 1.0, 0.90, 0.05)

# --- CALCULATIONS (Using Active Config) ---
# Calculate on the fly using the new LOG logic
survey_stats = calculate_dynamic_stats(filtered_df, ACTIVE_MODEL_CONFIG)
r2_cols = list(ACTIVE_MODEL_CONFIG.keys())

# 1. Calculate Best Model/Score
survey_stats["best_r2"] = survey_stats[r2_cols].max(axis=1)
survey_stats["winning_col"] = survey_stats[r2_cols].idxmax(axis=1)
survey_stats["winning_model"] = survey_stats["winning_col"].map(
    lambda x: ACTIVE_MODEL_CONFIG[x]["name"] if pd.notnull(x) else "None"
)

# 2. Enrich with Meta Data
meta_lookup = filtered_df[
    ["survey_id", "iso3", "year", "gdp_pcap", "gini_disp", "population"]
].drop_duplicates()
survey_stats = pd.merge(survey_stats, meta_lookup, on="survey_id", how="left")
survey_stats = survey_stats[survey_stats["best_r2"] > -100].copy()

# 3. Global Means for Sidebar Table
global_means = survey_stats[r2_cols].replace(-999.0, np.nan).mean().reset_index()
global_means.columns = ["model_key", "mean_r2"]
global_means["Model"] = global_means["model_key"].map(
    lambda x: ACTIVE_MODEL_CONFIG[x]["name"]
)
global_means = global_means.sort_values(by="mean_r2", ascending=False)

# 4. Labels & Logging
label_high = f"High (≥ {thresh_high:.2f})"
label_med = f"Medium (≥ {thresh_low:.2f} - < {thresh_high:.2f})"
label_low = f"Low (< {thresh_low:.2f})"


def get_quality_label(r2):
    if r2 >= thresh_high:
        return label_high
    elif r2 >= thresh_low:
        return label_med
    else:
        return label_low


survey_stats["Quality"] = survey_stats["best_r2"].apply(get_quality_label)
counts = survey_stats["Quality"].value_counts()
current_results = {
    "n_surveys": unique_surveys,
    "high": counts.get(label_high, 0),
    "med": counts.get(label_med, 0),
    "low": counts.get(label_low, 0),
    "label_high": label_high,
    "label_med": label_med,
    "label_low": label_low,
}
current_params = {
    "scale": scale_mode,
    "min_pop": min_pop_mil,
    "p_range": f"{min_p}-{max_p}",
    "t_low": thresh_low,
    "t_high": thresh_high,
    "models": len(selected_model_names),
}

# Change Log Logic
should_log = False
if not st.session_state.change_log:
    should_log = True
elif st.session_state.change_log[-1]["params"] != current_params:
    should_log = True

if should_log:
    deltas = {"n_surveys": 0, "high": 0, "med": 0, "low": 0}
    changes = ["Initial Load"]
    if st.session_state.change_log:
        last = st.session_state.change_log[-1]
        for k in deltas:
            deltas[k] = (
                current_results[
                    k.replace("n_surveys", "n_surveys") if k == "n_surveys" else k
                ]
                - last["results"][
                    k.replace("n_surveys", "n_surveys") if k == "n_surveys" else k
                ]
            )
        changes = [
            f"{k}: {last['params'][k]} → {v}"
            for k, v in current_params.items()
            if v != last["params"][k]
        ]

    st.session_state.change_log.append(
        {
            "id": len(st.session_state.change_log) + 1,
            "params": current_params,
            "results": current_results,
            "deltas": deltas,
            "changes_txt": ", ".join(changes),
        }
    )

# --- SORTING & SELECTION ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Sort Surveys By:**")

sort_option = st.sidebar.selectbox(
    "Sort Criteria",
    [
        "Best R² Score (Desc)",
        "Worst R² Score (Asc)",
        "ISO3 Code (A-Z)",
        "Year (Desc)",
        "Gini (High-Low)",
        "Gini (Low-High)",
        "GDP (High-Low)",
        "GDP (Low-High)",
    ],
    label_visibility="collapsed",
)

if sort_option == "Best R² Score (Desc)":
    survey_stats = survey_stats.sort_values(by="best_r2", ascending=False)
elif sort_option == "Worst R² Score (Asc)":
    survey_stats = survey_stats.sort_values(by="best_r2", ascending=True)
elif "ISO3" in sort_option:
    survey_stats = survey_stats.sort_values(
        by=["iso3", "year"], ascending=[True, False]
    )
elif "Year" in sort_option:
    survey_stats = survey_stats.sort_values(
        by=["year", "iso3"], ascending=[False, True]
    )
elif sort_option == "Gini (High-Low)":
    survey_stats = survey_stats.sort_values(by="gini_disp", ascending=False)
elif sort_option == "Gini (Low-High)":
    survey_stats = survey_stats.sort_values(by="gini_disp", ascending=True)
elif sort_option == "GDP (High-Low)":
    survey_stats = survey_stats.sort_values(by="gdp_pcap", ascending=False)
elif sort_option == "GDP (Low-High)":
    survey_stats = survey_stats.sort_values(by="gdp_pcap", ascending=True)

sorted_survey_list = survey_stats["survey_id"].tolist()
r2_lookup = dict(zip(survey_stats["survey_id"], survey_stats["best_r2"]))
gini_lookup = dict(zip(survey_stats["survey_id"], survey_stats["gini_disp"]))
gdp_lookup = dict(zip(survey_stats["survey_id"], survey_stats["gdp_pcap"]))

if "last_selected_survey" not in st.session_state:
    st.session_state.last_selected_survey = (
        sorted_survey_list[0] if sorted_survey_list else None
    )


def update_survey_selection():
    st.session_state.last_selected_survey = st.session_state.survey_widget


def format_survey_display(s_id):
    r2 = r2_lookup.get(s_id, 0)
    gini = gini_lookup.get(s_id, 0)
    gdp = gdp_lookup.get(s_id, 0)
    return f"{s_id} | R²: {r2:.2f} | Gini: {gini:.1f} | GDP: ${gdp:,.0f}"


try:
    pre_selected_index = sorted_survey_list.index(st.session_state.last_selected_survey)
except:
    pre_selected_index = 0

st.sidebar.markdown("---")
st.sidebar.markdown("**Select Survey (For Single View)**")
if sorted_survey_list:
    selected_survey = st.sidebar.selectbox(
        "Select Survey",
        options=sorted_survey_list,
        format_func=format_survey_display,
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


# --- PLOT GENERATION HELPER ---
def generate_survey_figure(survey_id, df_source):
    """Reusable function to generate the Plotly chart for a specific survey"""
    dff = df_source[df_source["survey_id"] == survey_id].copy()

    rename_map = {"true_daily_consumption": "TRUE INCOME"}
    # Only use columns from ACTIVE_MODEL_CONFIG
    for k, v in ACTIVE_MODEL_CONFIG.items():
        if v["pred_col"] in dff.columns:
            rename_map[v["pred_col"]] = v["name"]

    # Filter columns to only include keys in rename_map + percentile
    cols_to_keep = ["percentile"] + [c for c in rename_map.keys() if c in dff.columns]
    plot_df = (
        dff[cols_to_keep]
        .rename(columns=rename_map)
        .melt(
            id_vars=["percentile"],
            value_vars=[rename_map[c] for c in cols_to_keep if c != "percentile"],
            var_name="Model",
            value_name="Prediction",
        )
    )

    y_axis_label = "Log(Daily Consumption)" if is_log_y else "Daily Consumption ($)"
    x_axis_label = "Log(Percentile)" if is_log_x else "Percentile"
    if is_log_y:
        plot_df["Prediction"] = np.log(np.maximum(plot_df["Prediction"], 0.01))
    if is_log_x:
        plot_df["percentile"] = np.log(np.maximum(plot_df["percentile"], 0.01))

    fig = px.line(
        plot_df,
        x="percentile",
        y="Prediction",
        color="Model",
        color_discrete_map=COLOR_PALETTE,
        category_orders={"Model": MODEL_LOGICAL_ORDER},
        height=500,
        labels={"percentile": x_axis_label, "Prediction": y_axis_label},
        title=f"{survey_id}",
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_traces(
        selector={"name": "TRUE INCOME"},
        mode="markers",
        marker=dict(size=8, symbol="circle"),
    )
    if "r2_simulated" in ACTIVE_MODEL_CONFIG:
        fig.update_traces(
            selector={"name": ACTIVE_MODEL_CONFIG["r2_simulated"]["name"]},
            line=dict(width=3, dash="dash"),
        )
    return fig


# --- ACTION CALLBACK: GO TO SURVEY ---
def go_to_survey(s_id):
    st.session_state.survey_widget = s_id
    st.session_state.last_selected_survey = s_id
    st.session_state.active_tab = "📈 Chart Visualization"


# 5. MAIN CONTENT
st.title(f"Analysis: {scale_mode} Scale")

tab_options = [
    "📈 Chart Visualization",
    "📉 Chart Feed",
    "📊 Model Analysis",
    "🌍 Survey Scatter",
    "🗺️ Coverage Map",
    "📝 Model Equations",
    "📜 Change Log",
]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🌍 Survey Scatter"

selected_tab = st.radio(
    "Select View",
    options=tab_options,
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)
st.markdown("---")


def get_map_dataframe(
    pop_threshold, lmic_isos, gdf_admin0, df_names, raw_pop_df, cons_df
):
    cons_isos = set(cons_df["iso3"].astype(str).str.strip().str.upper())
    year_cols = [c for c in raw_pop_df.columns if c.isdigit()]
    latest_year = year_cols[-1]
    df_pop = raw_pop_df[["Country Code", latest_year]].copy()
    df_pop.columns = ["iso3", "population"]

    gdf_map = gdf_admin0.copy()
    if "population" in gdf_map.columns:
        gdf_map = gdf_map.drop(columns=["population"])
    if "country_name" in gdf_map.columns:
        gdf_map = gdf_map.drop(columns=["country_name"])

    gdf_map = gdf_map.merge(df_pop, on="iso3", how="left")
    gdf_map = gdf_map.merge(df_names, on="iso3", how="left")
    gdf_map["country_name"] = gdf_map["country_name"].fillna(gdf_map["iso3"])

    def get_status(row):
        iso = str(row["iso3"]).strip().upper() if row["iso3"] else ""
        in_lmic = iso in lmic_isos
        in_cons = iso in cons_isos
        pop = row["population"] if pd.notnull(row["population"]) else 0

        if not in_lmic and not in_cons:
            return "Excluded (High Income)"
        if in_lmic:
            if in_cons:
                return (
                    "Excluded (Small Pop)"
                    if pop < pop_threshold
                    else "Covered (Data and LMIC)"
                )
            else:
                return "Missing Data (LMIC)"
        return "Non-LMIC with Data"

    gdf_map["Status"] = gdf_map.apply(get_status, axis=1)
    lmic_mask = gdf_map["iso3"].apply(lambda x: str(x).upper() in lmic_isos)
    total_lmic_pop = gdf_map.loc[lmic_mask, "population"].sum()
    valid_mask = gdf_map["Status"] == "Covered (Data and LMIC)"
    covered_valid_pop = gdf_map.loc[valid_mask, "population"].sum()
    pct = (covered_valid_pop / total_lmic_pop * 100) if total_lmic_pop > 0 else 0
    return gdf_map, pct, covered_valid_pop, total_lmic_pop


# --- VIEW 1: SINGLE CHART ---
if selected_tab == "📈 Chart Visualization":
    survey_row = survey_stats[survey_stats["survey_id"] == selected_survey].iloc[0]

    current_pop = survey_row["population"]
    current_gini = survey_row["gini_disp"]
    current_gdp = survey_row["gdp_pcap"]

    st.markdown(f"### Survey: {selected_survey}")

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Population",
        f"{current_pop/1_000_000:.1f}M" if pd.notnull(current_pop) else "N/A",
    )
    m2.metric(
        "Gini Coefficient", f"{current_gini:.1f}" if pd.notnull(current_gini) else "N/A"
    )
    m3.metric(
        "GDP per Capita", f"${current_gdp:,.0f}" if pd.notnull(current_gdp) else "N/A"
    )

    with st.expander(f"Show {scale_mode} Scores (Active Models)", expanded=True):
        # UPDATED: Show R2 and RMSE (Log Scale)
        score_data = []
        for k in r2_cols:
            score_data.append(
                {
                    "Model Name": ACTIVE_MODEL_CONFIG[k]["name"],
                    "Log R²": survey_row[k],
                    "Log RMSE": survey_row.get(f"{k}_rmse", 999.0),
                }
            )

        st.dataframe(
            pd.DataFrame(score_data)
            .sort_values(by="Log R²", ascending=False)
            .style.background_gradient(subset=["Log R²"], cmap="Greens")
            .background_gradient(subset=["Log RMSE"], cmap="Reds")
            .format({"Log R²": "{:.3f}", "Log RMSE": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # Use the reusable function
    fig = generate_survey_figure(selected_survey, filtered_df)
    st.plotly_chart(fig, use_container_width=True)

# --- VIEW 1.5: CHART FEED (NEW) ---
elif selected_tab == "📉 Chart Feed":
    st.markdown("### 📉 All Charts Feed")
    st.caption("Scroll down to view charts. Use pagination to load more.")

    # Pagination controls
    items_per_page = 100
    total_items = len(sorted_survey_list)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

    col_page, col_info = st.columns([1, 4])
    with col_page:
        page_number = st.number_input(
            "Page", min_value=1, max_value=total_pages, value=1
        )
    with col_info:
        st.write(
            f"Showing page {page_number} of {total_pages} ({total_items} total surveys)"
        )

    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    current_batch = sorted_survey_list[start_idx:end_idx]

    # Render loop
    for s_id in current_batch:
        # Get stats for this specific survey from the pre-calculated stats df
        s_stats = survey_stats[survey_stats["survey_id"] == s_id].iloc[0]

        # Header Info with Button
        col_header, col_btn = st.columns([4, 1])
        with col_header:
            st.markdown(f"#### {s_id}")
        with col_btn:
            # BUTTON: switches tab and loads survey
            st.button(
                "View Details 🔍",
                key=f"btn_{s_id}",
                on_click=go_to_survey,
                args=(s_id,),
            )

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.caption(f"**Best Model:** {s_stats['winning_model']}")
        col_m2.caption(f"**Best R²:** {s_stats['best_r2']:.3f}")
        col_m3.caption(
            f"**Gini:** {s_stats['gini_disp']:.1f} | **GDP:** ${s_stats['gdp_pcap']:,.0f}"
        )

        # Chart
        fig = generate_survey_figure(s_id, filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

# --- VIEW 2: MODEL ANALYSIS ---
elif selected_tab == "📊 Model Analysis":
    st.markdown(f"### Winning Models Distribution ({scale_mode})")
    win_counts = survey_stats["winning_model"].value_counts().reset_index()
    win_counts.columns = ["Model", "Count"]
    win_counts["Percentage"] = (win_counts["Count"] / win_counts["Count"].sum()) * 100

    fig_bar = px.bar(
        win_counts,
        x="Model",
        y="Percentage",
        text_auto=".2f",
        color="Model",
        color_discrete_map=COLOR_PALETTE,
    )
    fig_bar.update_layout(yaxis_title="Percentage Won (%)", showlegend=True)
    fig_bar.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 🏆 Survey Performance")
    sort_order = st.radio(
        "Sort Order:",
        options=["Best to Worst (High -> Low)", "Worst to Best (Low -> High)"],
        horizontal=True,
    )
    chart_df = survey_stats.sort_values(
        by="best_r2", ascending=("Worst" in sort_order)
    ).reset_index(drop=True)
    chart_df["visual_r2"] = np.maximum(chart_df["best_r2"], -0.1)

    fig_scroll = px.bar(
        chart_df,
        x=chart_df.index,
        y="visual_r2",
        color="winning_model",
        color_discrete_map=COLOR_PALETTE,
        category_orders={"winning_model": MODEL_LOGICAL_ORDER},
        hover_name="survey_id",
        width=max(1000, len(chart_df) * 15),
        height=600,
        title=f"All Surveys ({len(chart_df)})",
    )
    fig_scroll.update_layout(
        xaxis_title="Survey Count (Index)",
        yaxis_title="Best R² Score",
        legend_title="Winning Model",
    )
    st.plotly_chart(fig_scroll, use_container_width=False)

# --- VIEW 3: SCATTER OVERVIEW ---
elif selected_tab == "🌍 Survey Scatter":
    st.markdown("### 🌍 Global Performance Overview")

    # 1. Prepare Base Data
    scatter_df = survey_stats.copy()
    scatter_df["pop_m"] = scatter_df["population"] / 1_000_000

    color_map_qual = {
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
        color_discrete_map=color_map_qual,
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
    st.markdown("### 🔬 Detailed Fit Analysis")

    with st.expander("⚙️ Customize Bucket Sizes", expanded=False):
        c_bin1, c_bin2 = st.columns(2)
        with c_bin1:
            gini_step = st.number_input(
                "Gini Interval Size", min_value=1, max_value=20, value=5
            )
        with c_bin2:
            gdp_step = st.select_slider(
                "GDP Interval Size ($)",
                options=[1000, 2000, 2500, 5000, 10000],
                value=2000,
            )

    gini_bins = list(range(0, 101, gini_step))
    gini_labels = [f"{i}-{i+gini_step}" for i in gini_bins[:-1]]

    scatter_df["gini_group"] = pd.cut(
        scatter_df["gini_disp"],
        bins=gini_bins + [200],
        labels=gini_labels + [f"> {gini_bins[-1]}"],
        right=False,
    )

    max_gdp = scatter_df["gdp_pcap"].max()
    gdp_bins = list(range(0, int(max_gdp) + gdp_step + 1, gdp_step))
    gdp_labels = [f"{int(i/1000)}k-{int((i+gdp_step)/1000)}k" for i in gdp_bins[:-1]]

    scatter_df["gdp_group"] = pd.cut(
        scatter_df["gdp_pcap"], bins=gdp_bins, labels=gdp_labels, right=False
    )

    def plot_grouped_distribution(group_col, title_label):
        df_grouped = (
            scatter_df.groupby([group_col, "Quality"], observed=False)
            .size()
            .reset_index(name="Count")
        )
        df_grouped = df_grouped[df_grouped["Count"] > 0]

        fig = px.bar(
            df_grouped,
            x=group_col,
            y="Count",
            color="Quality",
            text="Count",
            color_discrete_map=color_map_qual,
            category_orders={"Quality": [label_high, label_med, label_low]},
            title=f"Fit Count by {title_label} (Bucket Size: {gini_step if 'Gini' in title_label else gdp_step})",
            height=500,
            barmode="group",
        )
        fig.update_layout(
            xaxis_title=title_label,
            yaxis_title="Number of Surveys",
            template="plotly_white",
        )
        fig.update_traces(textposition="auto")
        return fig

    def plot_detailed_strip(group_col, x_label):
        counts = scatter_df[group_col].value_counts()

        def fmt_label(val):
            return f"{val}<br>(n={counts.get(val, 0)})"

        scatter_df[f"{group_col}_label"] = scatter_df[group_col].apply(fmt_label)
        cat_order = scatter_df[group_col].cat.categories
        sorted_cats = [fmt_label(l) for l in cat_order if counts.get(l, 0) > 0]

        fig = px.strip(
            scatter_df,
            x=f"{group_col}_label",
            y="best_r2",
            color="winning_model",
            color_discrete_map=COLOR_PALETTE,
            category_orders={
                f"{group_col}_label": sorted_cats,
                "winning_model": MODEL_LOGICAL_ORDER,
            },
            hover_name="survey_id",
            hover_data={"iso3": True, "population": ":,.0f"},
            stripmode="overlay",
            height=550,
            title=f"Individual Scores by {x_label}",
        )
        fig.update_yaxes(range=[0, 1.02])
        fig.add_hline(y=thresh_high, line_dash="dot", line_color="green", opacity=0.5)
        fig.add_hline(y=thresh_low, line_dash="dot", line_color="orange", opacity=0.5)
        fig.update_layout(
            template="plotly_white", xaxis_title=x_label, yaxis_title="Best R² Score"
        )
        return fig

    def plot_bubble_grid():
        grid_data = (
            scatter_df.groupby(["gini_group", "gdp_group"], observed=False)
            .agg(mean_r2=("best_r2", "mean"), count=("best_r2", "count"))
            .reset_index()
        )
        grid_data = grid_data[grid_data["count"] > 0]

        fig = px.scatter(
            grid_data,
            x="gdp_group",
            y="gini_group",
            size="count",
            color="mean_r2",
            color_continuous_scale="RdYlGn",
            range_color=[0.5, 1.0],
            size_max=40,
            hover_data={"count": True, "mean_r2": ":.2f"},
            title="Bubble Grid: Size = Survey Count, Color = Mean R²",
            height=600,
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="GDP Bracket",
            yaxis_title="Gini Bracket",
            xaxis=dict(showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=True, gridcolor="#eee"),
        )
        fig.update_traces(text=grid_data["count"], textposition="middle center")
        return fig

    tab_gini, tab_gdp, tab_matrix = st.tabs(
        ["By Gini", "By GDP", "Combined Bubble Grid"]
    )

    with tab_gini:
        chart_type_gini = st.radio(
            "Chart Type:",
            ["Grouped Bar (Summary)", "Detailed Dots"],
            key="gini_toggle",
            horizontal=True,
        )
        if chart_type_gini == "Grouped Bar (Summary)":
            st.plotly_chart(
                plot_grouped_distribution("gini_group", "Gini Bracket"),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                plot_detailed_strip("gini_group", "Gini Bracket"),
                use_container_width=True,
            )

    with tab_gdp:
        chart_type_gdp = st.radio(
            "Chart Type:",
            ["Grouped Bar (Summary)", "Detailed Dots"],
            key="gdp_toggle",
            horizontal=True,
        )
        if chart_type_gdp == "Grouped Bar (Summary)":
            st.plotly_chart(
                plot_grouped_distribution("gdp_group", "GDP Bracket"),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                plot_detailed_strip("gdp_group", "GDP Bracket"),
                use_container_width=True,
            )

    with tab_matrix:
        st.caption("Larger bubbles = More surveys. Green = Better fit.")
        st.plotly_chart(plot_bubble_grid(), use_container_width=True)

    st.markdown("---")

    st.markdown("### 📊 Overall Fit Quality Distribution")
    quality_counts = scatter_df["Quality"].value_counts().reset_index()
    quality_counts.columns = ["Quality", "Count"]
    quality_counts["Percentage"] = (
        quality_counts["Count"] / quality_counts["Count"].sum()
    ) * 100

    fig_bar_qual = px.bar(
        quality_counts,
        x="Quality",
        y="Count",
        color="Quality",
        text_auto=True,
        color_discrete_map=color_map_qual,
        category_orders={"Quality": [label_high, label_med, label_low]},
        hover_data={"Percentage": ":.1f"},
        height=400,
    )
    fig_bar_qual.update_layout(
        yaxis_title="Number of Surveys", template="plotly_white", showlegend=False
    )
    fig_bar_qual.update_traces(texttemplate="%{y} <br>(%{customdata[0]:.1f}%)")
    st.plotly_chart(fig_bar_qual, use_container_width=True)

# --- VIEW 4: MAP ---
elif selected_tab == "🗺️ Coverage Map":
    st.markdown("### 🗺️ Data Coverage Map")
    lmic_isos, gdf_admin0, df_names = load_map_resources()
    if lmic_isos is None:
        st.error(f"❌ Failed to load map resources. Error: {df_names}")
    else:
        with st.spinner("Generating Map..."):
            gdf_map, pct, pop_cov, pop_tot = get_map_dataframe(
                min_pop_val, lmic_isos, gdf_admin0, df_names, raw_pop_df, raw_df
            )
            k1, k2, k3 = st.columns(3)
            k1.metric("Coverage (Pop)", f"{pct:.1f}%")
            k2.metric("Covered People", f"{pop_cov/1e9:.2f}B")
            k3.metric("Total LMIC Pop", f"{pop_tot/1e9:.2f}B")

            color_map = {
                "Covered (Data and LMIC)": "#2ca02c",
                "Excluded (Small Pop)": "#ffd700",
                "Missing Data (LMIC)": "#d62728",
                "Non-LMIC with Data": "#1f77b4",
                "Excluded (High Income)": "#eeeeee",
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
    st.info(
        "ℹ️ **Notation:** $Y_{p,c,y}$ = Consumption, $P$ = Percentile Rank. Subscripts: $p$ (percentile), $c$ (country), $y$ (year)."
    )

    Y_term = r"\ln(Y_{p,c,y})" if is_log_y else "Y_{p,c,y}"
    P_term = r"\ln(P)" if is_log_x else "P"

    st.markdown("---")
    st.subheader("0. Simulated")
    c1, c2 = st.columns([1, 1])
    c1.latex(r"Y \sim \text{LogNormal}(\mu_{c,y}, \sigma_{c,y})")
    c2.write("Simulated using Log-Normal distribution from summary stats.")

    st.markdown("---")
    st.subheader("1. Global Fixed Slope")
    c1, c2 = st.columns(2)
    c1.latex(
        f"{Y_term} = \\beta_0 + \\beta_1 \\text{{GDP}}_{{c,y}} + \\beta_2 {P_term} + \\epsilon"
    )
    c2.code("feols(Y ~ gdp + pct, cluster = ~iso3)", language="r")

    st.markdown("---")
    st.subheader("2. Gini Interaction")
    c1, c2 = st.columns(2)
    c1.latex(
        f"{Y_term} = \\beta_0 + \\beta_1 \\text{{GDP}} + \\beta_2 \\text{{Gini}} + \\beta_3 ({P_term} \\times \\text{{Gini}}) + \\epsilon"
    )
    c2.code("feols(Y ~ gdp + gini + pct * gini, cluster=~iso3)", language="r")

    st.markdown("---")
    st.subheader("3. Country-Specific Trends")
    c1, c2 = st.columns(2)
    c1.latex(
        f"{Y_term} = \\alpha_c + \\beta_1 \\text{{GDP}} + \\beta_{{2,c}} {P_term} + \\epsilon"
    )
    c2.code("feols(Y ~ gdp | iso3[pct])", language="r")

    st.markdown("---")
    st.subheader("4-7. Survey Fits")
    st.write("Standard polynomial regressions on the percentile.")
    c1, c2 = st.columns(2)
    c1.latex(
        f"{Y_term} = \\beta_0 + \\sum_{{k=1}}^{{d}} \\beta_k ({P_term})^k + \\epsilon"
    )
    c2.code("lm(Y ~ poly(pct, degree))", language="r")

    st.markdown("---")
    st.subheader("8. Segmented (Piecewise)")
    st.write("Linear slope in the middle (16-84), Quadratic curves in the tails.")
    c1, c2 = st.columns(2)
    c1.latex(
        r"Y = \begin{cases} \beta_0 + \beta_1 P + \beta_2 P^2 & \text{if } P \in \text{Tails} \\ \alpha_0 + \alpha_1 P & \text{if } P \in \text{Middle} \end{cases}"
    )
    c2.code("feols(Y ~ gdp + gini:mid + gini:tail + gini:tail^2)", language="r")


# --- VIEW 6: LOG ---
elif selected_tab == "📜 Change Log":
    st.markdown("### 📜 Session History")
    if not st.session_state.change_log:
        st.info("No changes recorded yet.")
    else:
        for i, entry in enumerate(reversed(st.session_state.change_log)):
            res, delta = entry["results"], entry["deltas"]
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
