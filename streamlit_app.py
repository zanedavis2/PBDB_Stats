import streamlit as st
import pandas as pd
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="PBDB Stats Dashboard", layout="wide")

# --- DATA PROCESSING FUNCTIONS FROM SOURCE ---

def clean_df(df):
    """Standardizes column names and removes empty/total rows."""
    if "Last" in df.columns and "First" in df.columns:
        df["Last"] = df["Last"].astype(str).str.strip()
        df["First"] = df["First"].astype(str).str.strip()
        df["Last"].replace(["", "nan", "NaN", "None"], np.nan, inplace=True)
        df["First"].replace(["", "nan", "NaN", "None"], np.nan, inplace=True)
        totals_idx = df.index[df["Last"].isna() & df["First"].isna()]
        if len(totals_idx) > 0:
            df = df.loc[:totals_idx[0] - 1].reset_index(drop=True)
    return df

# Batting
def prepare_batting_stats(df):
    df = df.copy()
    columns_to_keep = ["Last", "First", "PA", "AB", "H", "AVG", "OBP", "SLG", "OPS", "RBI", "R", "BB", "SO", "XBH", "2B", "3B","HR", "TB", "SB", "PS/PA", "BB/K", "C%", "QAB", "QAB%", "HHB", "HHB %", "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "2OUTRBI"]
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()
    if "PA" in df.columns:
        df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
        df = df[df["PA"] != 0].reset_index(drop=True)
    return df.sort_values(by=["Last", "First"])

# Pitching
def prepare_pitching_stats(df):
    df = df.copy()
    columns_to_keep = ["Last", "First", "IP", "ERA", "WHIP", "H", "R", "ER", "BB", "BB/INN", "SO", "K-L", "HR", "S%", "FPS%", "FPSO%", "FPSH%", "SM%", "<3%", "LD%", "FB%", "GB%", "HHB%", "WEAK%", "BBS", "BAA", "BABIP", "BA/RISP", "CS", "SB", "SB%", "FIP"]
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()
    if "IP" in df.columns:
        df["IP"] = pd.to_numeric(df["IP"], errors="coerce")
        df = df[df["IP"] != 0].reset_index(drop=True)
    for col in df.columns:
        if col not in ["Last", "First", "BABIP", "BAA", "BA/RISP"] and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(2)
    return df.sort_values(by=["Last", "First"])

# Fielding
def prepare_fielding_stats(df):
    df = df.copy()
    columns_to_keep = ["Last", "First", "TC", "A", "PO", "FPCT", "E", "DP"]
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()
    if "TC" in df.columns:
        df["TC"] = pd.to_numeric(df["TC"], errors="coerce")
        df = df[df["TC"] != 0].reset_index(drop=True)
    for col in df.columns:
        if col not in ["Last", "First","FPCT"] and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(0)
    return df.sort_values(by=["Last", "First"])

# Catching
def prepare_catching_stats(df):
    df = df.copy()
    columns_to_keep = ["Last", "First", "INN", "PB", "SB-ATT", "CS", "CS%"]
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()
    if "INN" in df.columns:
        df["INN"] = pd.to_numeric(df["INN"], errors="coerce")
        df = df[df["INN"] != 0].reset_index(drop=True)
    return df.sort_values(by=["Last", "First"])

# --- APP UI ---

st.title("⚾ PBDB Baseball Statistics")
st.markdown("Analyze performance using cumulative data or aggregate specific series from the repository.")

# Sidebar for Mode and Filters
st.sidebar.header("Navigation")
mode = st.sidebar.radio("View Mode", ["Cumulative Stats", "Series Aggregator"])
category = st.sidebar.selectbox("Stat Category", ["Hitting", "Pitching", "Fielding", "Catching"])

# Available series files in the repo (assuming these filenames exist)
available_files = ["wake", "jmu", "unc"]

if mode == "Cumulative Stats":
    file_path = "cumulative.csv"
    if os.path.exists(file_path):
        raw_df = pd.read_csv(file_path, header=1)
        base_df = clean_df(raw_df)
    else:
        st.error(f"'{file_path}' not found in the repository.")
        st.stop()

else:
    # Aggregator Logic based on specific multi-file functions
    selected_series = st.sidebar.multiselect("Select Series to Combine", available_files, default=["wake"])
    
    if not selected_series:
        st.warning("Please select at least one series.")
        st.stop()
        
    # Import aggregation logic from your script
    from pbdb_stats import aggregate_stats_hitting, generate_aggregated_hitting_df, aggregate_stats_pitching, generate_aggregated_pitching_df, aggregate_stats_fielding, aggregate_stats_catching
    
    if category == "Hitting":
        base_df = generate_aggregated_hitting_df(aggregate_stats_hitting(selected_series))
    elif category == "Pitching":
        base_df = generate_aggregated_pitching_df(aggregate_stats_pitching(selected_series))
    elif category == "Fielding":
        base_df = aggregate_stats_fielding(selected_series)
    else:
        base_df = aggregate_stats_catching(selected_series)

# Player Filtering
all_players = sorted(base_df["Last"].dropna().unique())
player_filter = st.sidebar.multiselect("Filter by Player", all_players)

# Final Preparation based on Category
if category == "Hitting":
    final_df = prepare_batting_stats(base_df)
elif category == "Pitching":
    final_df = prepare_pitching_stats(base_df)
elif category == "Fielding":
    final_df = prepare_fielding_stats(base_df)
else:
    final_df = prepare_catching_stats(base_df)

if player_filter:
    final_df = final_df[final_df["Last"].isin(player_filter)]

# Display Data
st.subheader(f"{category} - {mode}")
st.dataframe(final_df, use_container_width=True, hide_index=True)

# Acronym Cheat Sheet
with st.expander("📖 Stat Definition Reference"):
    if category == "Hitting":
        data = {"Acronym": ["PA", "AB", "H", "AVG", "OBP", "SLG", "OPS", "RBI", "QAB%", "BABIP"], 
                "Meaning": ["Plate Appearances", "At-Bats", "Hits", "Batting Average", "On-Base Percentage", "Slugging Percentage", "On-base Plus Slugging", "Runs Batted In", "Quality At-Bat Percentage", "Batting Average on Balls In Play"]}
    elif category == "Pitching":
        data = {"Acronym": ["IP", "ERA", "WHIP", "FPS%", "SM%", "BAA", "FIP"], 
                "Meaning": ["Innings Pitched", "Earned Run Average", "Walks plus Hits per IP", "First-Pitch Strike Percentage", "Swinging Miss Percentage", "Batting Average Against", "Fielding Independent Pitching"]}
    else:
        data = {"Acronym": ["FPCT", "TC", "CS%", "PB"], "Meaning": ["Fielding Percentage", "Total Chances", "Caught Stealing Percentage", "Passed Balls"]}
    st.table(pd.DataFrame(data))
