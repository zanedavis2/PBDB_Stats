"""
EUCB Baseball Stats Dashboard
A streamlined application for viewing and analyzing baseball statistics.
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# CONFIGURATION
# ============================================================================

STAT_CATEGORIES = ["Hitting", "Pitching", "Fielding", "Catching"]

# Minimum qualifications for leaderboards
MIN_QUALIFICATIONS = {
    "Hitting": {"PA": 1},
    "Pitching": {"IP": 0.1},
    "Fielding": {"TC": 1},
    "Catching": {"INN": 0.1}
}

# Column definitions for each stat category
HITTING_COLS = [
    "Last", "First", "PA", "AB", "H", "AVG", "OBP", "SLG", "OPS", 
    "RBI", "R", "BB", "SO", "2B", "3B", "HR", "TB", "SB", "XBH",
    "PS/PA", "BB/K", "C%", "QAB", "QAB%", "HHB", "HHB%",
    "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "2OUTRBI"
]

PITCHING_COLS = [
    "Last", "First", "IP", "ERA", "WHIP", "H", "R", "ER", "BB", 
    "BB/INN", "SO", "K-L", "HR", "S%", "FPS%", "FPSO%", "FPSH%",
    "SM%", "<3%", "LD%", "FB%", "GB%", "HHB%", "WEAK%", 
    "BBS", "BAA", "BABIP", "BA/RISP", "CS", "SB", "SB%"
]

FIELDING_COLS = ["Last", "First", "TC", "A", "PO", "FPCT", "E", "DP"]

CATCHING_COLS = ["Last", "First", "INN", "PB", "SB-ATT", "CS", "CS%"]

# Stat explanations
STAT_KEYS = {
    "Hitting": pd.DataFrame({
        "Acronym": ["PA", "AB", "H", "AVG", "OBP", "SLG", "OPS", "RBI", "R", "BB", "SO", 
                    "XBH", "2B", "3B", "HR", "TB", "SB", "PS/PA", "BB/K", "C%", "QAB", 
                    "QAB%", "HHB", "HHB%", "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "2OUTRBI"],
        "Meaning": ["Plate Appearances", "At-Bats", "Hits", "Batting Average", 
                    "On-Base Percentage", "Slugging Percentage", "On-base Plus Slugging",
                    "Runs Batted In", "Runs Scored", "Walks", "Strikeouts", 
                    "Extra-Base Hits", "Doubles", "Triples", "Home Runs", "Total Bases",
                    "Stolen Bases", "Pitches per PA", "Walk-to-Strikeout Ratio",
                    "Contact %", "Quality At-Bats", "Quality At-Bat %", "Hard-Hit Balls",
                    "Hard-Hit Ball %", "Line Drive %", "Fly Ball %", "Ground Ball %",
                    "Batting Avg on Balls In Play", "Avg w/ RISP", "Two-Out RBIs"]
    }),
    "Pitching": pd.DataFrame({
        "Acronym": ["IP", "ERA", "WHIP", "H", "R", "ER", "BB", "BB/INN", "SO", "K-L", 
                    "HR", "S%", "FPS%", "FPSO%", "FPSH%", "SM%", "<3%", "LD%", "FB%", 
                    "GB%", "HHB%", "WEAK%", "BBS", "BAA", "BABIP", "BA/RISP", "CS", 
                    "SB", "SB%"],
        "Meaning": ["Innings Pitched", "Earned Run Average", "Walks + Hits per Inning",
                    "Hits Allowed", "Runs Allowed", "Earned Runs", "Walks", 
                    "Walks per Inning", "Strikeouts", "Strikeouts Looking", 
                    "Home Runs Allowed", "Strike %", "First-Pitch Strike %",
                    "% of FPS ABs ending in outs", "% of FPS that are hits",
                    "Swinging Miss %", "% of ABs ≤3 pitches", "Line Drive %",
                    "Fly Ball %", "Ground Ball %", "Hard-Hit Ball %", "Weak Contact %",
                    "BB resulting in run", "Batting Avg Against", "BABIP",
                    "Avg w/ RISP", "Caught Stealing", "Stolen Bases Allowed", "SB %"]
    }),
    "Fielding": pd.DataFrame({
        "Acronym": ["TC", "A", "PO", "FPCT", "E", "DP"],
        "Meaning": ["Total Chances", "Assists", "Putouts", "Fielding %", "Errors", 
                    "Double Plays"]
    }),
    "Catching": pd.DataFrame({
        "Acronym": ["INN", "PB", "SB-ATT", "CS", "CS%"],
        "Meaning": ["Innings Caught", "Passed Balls", "SB Attempts", "Caught Stealing", 
                    "CS %"]
    })
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_series():
    """Find all available series CSV files (excluding cumulative)."""
    all_csvs = glob.glob("*.csv")
    series = [os.path.splitext(f)[0] for f in all_csvs 
              if f.lower() != "cumulative.csv"]
    return sorted(series)


def clean_dataframe(df):
    """Remove invalid rows and normalize names."""
    if "Last" in df.columns and "First" in df.columns:
        df["Last"] = df["Last"].astype(str).str.strip().str.title()
        df["First"] = df["First"].astype(str).str.strip().str.title()
        
        # Remove rows with missing names or totals rows
        df = df[df["Last"].notna() & (df["Last"] != "")]
        df = df[df["First"].notna() & (df["First"] != "")]
        df = df[~df["Last"].str.lower().str.contains("total", na=False)]
    
    return df.reset_index(drop=True)


def convert_ip_format(ip):
    """Convert innings pitched from decimal (4.2) to proper format (4.667)."""
    try:
        whole = int(ip)
        fraction = round((ip - whole) * 10)
        if fraction == 1:
            return whole + 1/3
        elif fraction == 2:
            return whole + 2/3
        return float(ip)
    except:
        return 0.0


def safe_divide(numerator, denominator, default=0):
    """Safely divide with default value for division by zero."""
    return np.where(denominator > 0, numerator / denominator, default)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_cumulative_data():
    """Load the cumulative season stats CSV."""
    try:
        df = pd.read_csv("cumulative.csv", header=1)
        df = clean_dataframe(df)
        return df
    except FileNotFoundError:
        st.error("❌ cumulative.csv not found in the current directory")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading cumulative.csv: {e}")
        return pd.DataFrame()


def load_series_data(series_names):
    """Load and aggregate data from multiple series CSVs."""
    dfs = []
    for series in series_names:
        try:
            df = pd.read_csv(f"{series}.csv", header=1)
            df = clean_dataframe(df)
            dfs.append(df)
        except FileNotFoundError:
            st.warning(f"⚠️ {series}.csv not found")
        except Exception as e:
            st.warning(f"⚠️ Error loading {series}.csv: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

# ============================================================================
# HITTING STATS PROCESSING
# ============================================================================

def process_hitting_stats(df, is_aggregated=False):
    """Process and calculate hitting statistics."""
    if df.empty:
        return df
    
    # Select and order columns
    available_cols = [c for c in HITTING_COLS if c in df.columns]
    df = df[available_cols].copy()
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ["Last", "First"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    if is_aggregated:
        # Aggregate by player
        group_cols = ["Last", "First"]
        df = df.groupby(group_cols, as_index=False).sum(numeric_only=True)
        
        # Calculate derived stats
        df["AVG"] = safe_divide(df["H"], df["AB"])
        df["OBP"] = safe_divide(
            df["H"] + df["BB"] + df.get("HBP", 0),
            df["AB"] + df["BB"] + df.get("HBP", 0) + df.get("SF", 0)
        )
        df["SLG"] = safe_divide(df.get("TB", 0), df["AB"])
        df["OPS"] = df["OBP"] + df["SLG"]
        df["QAB%"] = safe_divide(df.get("QAB", 0), df["PA"])
        df["BB/K"] = safe_divide(df["BB"], df["SO"], df["BB"])
        df["C%"] = 1 - safe_divide(df["SO"], df["AB"])
        df["HHB%"] = safe_divide(df.get("HHB", 0), df["AB"])
        df["BABIP"] = safe_divide(
            df["H"] - df.get("HR", 0),
            df["AB"] - df["SO"] - df.get("HR", 0) + df.get("SF", 0)
        )
        df["PS/PA"] = safe_divide(df.get("PS", 0), df["PA"])
        
        # Round percentages
        pct_cols = ["AVG", "OBP", "SLG", "OPS", "QAB%", "BB/K", "C%", 
                    "HHB%", "BABIP", "PS/PA"]
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].round(3)
    
    # Apply minimum qualifications and sort
    min_pa = MIN_QUALIFICATIONS["Hitting"]["PA"]
    df = df[df["PA"] >= min_pa].reset_index(drop=True)
    df = df.sort_values(["Last", "First"]).reset_index(drop=True)
    
    return df

# ============================================================================
# PITCHING STATS PROCESSING
# ============================================================================

def process_pitching_stats(df, is_aggregated=False):
    """Process and calculate pitching statistics."""
    if df.empty:
        return df
    
    # Select relevant columns (adjust range for pitching data location)
    if is_aggregated:
        # For series aggregation, we need the raw counting stats
        available_cols = [c for c in PITCHING_COLS + ["#P", "BF", "HBP"] 
                         if c in df.columns]
    else:
        # For cumulative, extract pitching columns (typically cols 53-148)
        if df.shape[1] > 53:
            df = df.iloc[:, [1, 2] + list(range(53, min(148, df.shape[1])))]
            df.columns = [c.replace(".1", "") for c in df.columns]
    
    df = df.copy()
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ["Last", "First"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Convert IP format
    if "IP" in df.columns:
        df["IP"] = df["IP"].apply(convert_ip_format)
    
    if is_aggregated:
        # Aggregate by player
        group_cols = ["Last", "First"]
        df = df.groupby(group_cols, as_index=False).sum(numeric_only=True)
        
        # Calculate derived stats
        df["ERA"] = safe_divide(df["ER"] * 9, df["IP"]).round(2)
        df["WHIP"] = safe_divide(df["BB"] + df["H"], df["IP"]).round(2)
        df["BB/INN"] = safe_divide(df["BB"], df["IP"]).round(2)
        df["S%"] = safe_divide(df.get("Strikes", 0), df.get("#P", 1)) * 100
        df["FPS%"] = safe_divide(df.get("FPS", 0), df.get("BF", 1)) * 100
        df["BAA"] = safe_divide(
            df["H"], 
            df.get("BF", 0) - df["BB"] - df.get("HBP", 0)
        ).round(3)
        df["BABIP"] = safe_divide(
            df["H"] - df.get("HR", 0),
            df.get("BF", 0) - df["SO"] - df.get("HR", 0) - df["BB"] - df.get("HBP", 0)
        ).round(3)
        df["SB%"] = safe_divide(df.get("SB", 0), df.get("SB", 0) + df.get("CS", 0)) * 100
    
    # Select final columns
    available_cols = [c for c in PITCHING_COLS if c in df.columns]
    df = df[available_cols].copy()
    
    # Apply minimum qualifications and sort
    min_ip = MIN_QUALIFICATIONS["Pitching"]["IP"]
    df = df[df["IP"] >= min_ip].reset_index(drop=True)
    df = df.sort_values(["Last", "First"]).reset_index(drop=True)
    
    return df

# ============================================================================
# FIELDING STATS PROCESSING
# ============================================================================

def process_fielding_stats(df, is_aggregated=False):
    """Process and calculate fielding statistics."""
    if df.empty:
        return df
    
    # Extract fielding columns (typically cols 148+)
    if not is_aggregated and df.shape[1] > 148:
        df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
        df.columns = [c.replace(".1", "") for c in df.columns]
    
    df = df.copy()
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ["Last", "First"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    if is_aggregated:
        # Aggregate by player
        group_cols = ["Last", "First"]
        df = df.groupby(group_cols, as_index=False).sum(numeric_only=True)
    
    # Calculate fielding percentage
    df["FPCT"] = safe_divide(df.get("A", 0) + df.get("PO", 0), df.get("TC", 1)).round(3)
    
    # Select final columns
    available_cols = [c for c in FIELDING_COLS if c in df.columns]
    df = df[available_cols].copy()
    
    # Apply minimum qualifications and sort
    min_tc = MIN_QUALIFICATIONS["Fielding"]["TC"]
    df = df[df.get("TC", 0) >= min_tc].reset_index(drop=True)
    df = df.sort_values(["Last", "First"]).reset_index(drop=True)
    
    return df

# ============================================================================
# CATCHING STATS PROCESSING
# ============================================================================

def process_catching_stats(df, is_aggregated=False):
    """Process and calculate catching statistics."""
    if df.empty:
        return df
    
    # Extract catching columns
    if not is_aggregated and df.shape[1] > 148:
        df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
        df.columns = [c.replace(".1", "").replace(".2", "") for c in df.columns]
    
    df = df.copy()
    
    # Handle SB-ATT format
    if "SB-ATT" in df.columns:
        split = df["SB-ATT"].astype(str).str.split("-", expand=True)
        if split.shape[1] >= 2:
            df["SB"] = pd.to_numeric(split[0], errors="coerce").fillna(0)
            df["ATT"] = pd.to_numeric(split[1], errors="coerce").fillna(0)
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ["Last", "First", "SB-ATT"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    if is_aggregated:
        # Aggregate by player
        group_cols = ["Last", "First"]
        df = df.groupby(group_cols, as_index=False).sum(numeric_only=True)
        
        # Recalculate SB-ATT and CS%
        df["CS%"] = safe_divide(df.get("CS", 0), df.get("ATT", 1)) * 100
        df["SB-ATT"] = (df.get("SB", 0).astype(int).astype(str) + "-" + 
                        df.get("ATT", 0).astype(int).astype(str))
    
    # Select final columns
    available_cols = [c for c in CATCHING_COLS if c in df.columns]
    df = df[available_cols].copy()
    
    # Apply minimum qualifications and sort
    min_inn = MIN_QUALIFICATIONS["Catching"]["INN"]
    df = df[df.get("INN", 0) >= min_inn].reset_index(drop=True)
    df = df.sort_values(["Last", "First"]).reset_index(drop=True)
    
    return df

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="EUCB Stats Dashboard", layout="wide")
    st.title("⚾ EUCB Baseball Stats Dashboard")
    st.markdown("*Fall 2025 Season*")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Filters")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Cumulative", "Series"],
            help="Cumulative: Season totals | Series: Select specific games"
        )
        
        # Series selection (if applicable)
        selected_series = []
        if data_source == "Series":
            available_series = get_available_series()
            if not available_series:
                st.error("No series CSV files found!")
                st.stop()
            
            selected_series = st.multiselect(
                "Select Series",
                available_series,
                default=available_series[:1] if available_series else []
            )
            
            if not selected_series:
                st.warning("Please select at least one series")
                st.stop()
        
        # Stat category selection
        st.markdown("---")
        selected_categories = st.multiselect(
            "Stat Categories",
            STAT_CATEGORIES,
            default=STAT_CATEGORIES,
            help="Choose which stats to display"
        )
        
        if not selected_categories:
            st.warning("Please select at least one category")
            st.stop()
    
    # Load data
    if data_source == "Cumulative":
        raw_df = load_cumulative_data()
        if raw_df.empty:
            st.stop()
    else:
        raw_df = load_series_data(selected_series)
        if raw_df.empty:
            st.error("No data loaded from selected series")
            st.stop()
    
    # Process each category
    is_aggregated = (data_source == "Series")
    
    stats_data = {}
    if "Hitting" in selected_categories:
        stats_data["Hitting"] = process_hitting_stats(raw_df, is_aggregated)
    if "Pitching" in selected_categories:
        stats_data["Pitching"] = process_pitching_stats(raw_df, is_aggregated)
    if "Fielding" in selected_categories:
        stats_data["Fielding"] = process_fielding_stats(raw_df, is_aggregated)
    if "Catching" in selected_categories:
        stats_data["Catching"] = process_catching_stats(raw_df, is_aggregated)
    
    # Display tabs
    tabs = st.tabs(selected_categories)
    
    for tab, category in zip(tabs, selected_categories):
        with tab:
            df = stats_data.get(category, pd.DataFrame())
            
            if df.empty:
                st.info(f"No {category.lower()} data available")
                continue
            
            # Display stats table
            st.subheader(f"{category} Statistics")
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Display stat key
            with st.expander(f"📊 {category} Stat Definitions"):
                st.dataframe(
                    STAT_KEYS[category], 
                    use_container_width=True, 
                    hide_index=True
                )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label=f"⬇️ Download {category} Data",
                data=csv,
                file_name=f"eucb_{category.lower()}_stats.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
