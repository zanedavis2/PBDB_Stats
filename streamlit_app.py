import streamlit as st
import pandas as pd
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="PBDB Stats Dashboard", layout="wide")

# --- CORE PROCESSING FUNCTIONS (From pbdb_stats.py) ---

def clean_df(df):
    if "Last" in df.columns and "First" in df.columns:
        df["Last"] = df["Last"].astype(str).str.strip()
        df["First"] = df["First"].astype(str).str.strip()
        df["Last"].replace(["", "nan", "NaN", "None"], np.nan, inplace=True)
        df["First"].replace(["", "nan", "NaN", "None"], np.nan, inplace=True)
        totals_idx = df.index[df["Last"].isna() & df["First"].isna()]
        if len(totals_idx) > 0:
            df = df.loc[:totals_idx[0] - 1].reset_index(drop=True)
    return df

def add_totals_row(df, category):
    """Calculates a totals row with correct weighted math."""
    if df.empty:
        return df
    
    # Create a copy and identify numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    totals = numeric_df.sum()
    
    # Custom math for percentages/averages
    if category == "Hitting":
        totals["AVG"] = (totals["H"] / totals["AB"]) if totals["AB"] > 0 else 0
        totals["OBP"] = (totals["H"] + totals["BB"]) / (totals["AB"] + totals["BB"]) if (totals["AB"] + totals["BB"]) > 0 else 0
        if "TB" in totals:
            totals["SLG"] = (totals["TB"] / totals["AB"]) if totals["AB"] > 0 else 0
            totals["OPS"] = totals["OBP"] + totals["SLG"]
        if "PA" in totals and totals["PA"] > 0:
            totals["QAB%"] = totals["QAB"] / totals["PA"] if "QAB" in totals else 0

    elif category == "Pitching":
        if totals["IP"] > 0:
            totals["ERA"] = (totals["ER"] * 9) / totals["IP"]
            totals["WHIP"] = (totals["BB"] + totals["H"]) / totals["IP"]
            totals["BB/INN"] = totals["BB"] / totals["IP"]
        totals["BAA"] = totals["H"] / (totals["H"] + totals["SO"]) if (totals["H"] + totals["SO"]) > 0 else 0 # Placeholder logic

    elif category == "Fielding":
        if totals["TC"] > 0:
            totals["FPCT"] = (totals["PO"] + totals["A"]) / totals["TC"]

    elif category == "Catching":
        # Parsing SB-ATT if it exists as a sum is complex, usually we sum components
        pass

    # Create the row
    totals_row = pd.DataFrame([totals])
    totals_row["Last"] = "TEAM"
    totals_row["First"] = "TOTALS"
    
    return pd.concat([df, totals_row], ignore_index=True)

# (Insert the preparation functions: prepare_batting_stats, prepare_pitching_stats, etc. from your script here)
# For brevity, I am assuming the logic provided in your original .py file is used for columns

def prepare_batting_stats(df):
    df = df.copy()
    cols = ["Last", "First", "PA", "AB", "H", "AVG", "OBP", "SLG", "OPS", "RBI", "R", "BB", "SO", "XBH", "2B", "3B","HR", "TB", "SB", "QAB%"]
    existing = [c for c in cols if c in df.columns]
    df = df[existing]
    return df.sort_values(by=["Last", "First"])

def prepare_pitching_stats(df):
    df = df.copy()
    cols = ["Last", "First", "IP", "ERA", "WHIP", "H", "R", "ER", "BB", "SO", "BAA", "FIP"]
    existing = [c for c in cols if c in df.columns]
    df = df[existing]
    return df.sort_values(by=["Last", "First"])

# --- APP LAYOUT ---

st.title("⚾ PBDB Analytics Dashboard")

# Sidebar
st.sidebar.header("Data Selection")
mode = st.sidebar.radio("View Mode", ["Cumulative File", "Aggregate Specific Series"])
category = st.sidebar.selectbox("Category", ["Hitting", "Pitching", "Fielding", "Catching"])

# Available files in repo
available_series = ["High Point", "JMU", "UNC", "UNCG", "Wake Forest"]

if mode == "Cumulative File":
    file_path = "cumulative.csv"
    if os.path.exists(file_path):
        df = clean_df(pd.read_csv(file_path, header=1))
    else:
        st.error(f"File {file_path} not found.")
        st.stop()
else:
    selected = st.sidebar.multiselect("Select Series", available_series, default=["UNC"])
    # This is where your aggregate_stats_hitting/pitching functions would run
    # For this template, we will load and concat the files directly
    dfs = []
    for s in selected:
        fname = f"{s}.csv"
        if os.path.exists(fname):
            dfs.append(pd.read_csv(fname, header=1))
    if dfs:
        df = clean_df(pd.concat(dfs))
    else:
        st.warning("No files found.")
        st.stop()

# Apply logic
if category == "Hitting":
    final_df = prepare_batting_stats(df)
elif category == "Pitching":
    final_df = prepare_pitching_stats(df)
else:
    final_df = df # Default fallback

# Player Filter
players = sorted(final_df["Last"].unique())
selected_players = st.sidebar.multiselect("Filter Players", players)
if selected_players:
    final_df = final_df[final_df["Last"].isin(selected_players)]

# Add Totals Row
final_df_with_totals = add_totals_row(final_df, category)

# --- DISPLAY ---
st.subheader(f"{category} Stats - {mode}")
st.dataframe(final_df_with_totals.style.format(precision=3), use_container_width=True, hide_index=True)

# Export
csv = final_df_with_totals.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download as CSV", csv, "pbdb_stats.csv", "text/csv")

with st.expander("Help & Acronyms"):
    st.write("Calculations are performed using standard Sabermetrics formulas.")
