# eucb_stats_app.py
import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

STAT_TYPES_ALL = ["Hitting", "Pitching", "Fielding", "Catching"]
QUAL_MINS = {"Hitting": 1, "Pitching": 0.1, "Fielding": 1, "Catching": 0.1}

HITTING_KEY = pd.DataFrame({
    "Acronym": [
        "PA","AB","H","AVG","OBP","SLG","OPS","RBI","R","BB","HBP","SO","XBH","2B","3B","HR","TB","SB",
        "PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB %","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI"
    ],
    "Meaning": [
        "Plate Appearances","At-Bats","Hits","Batting Average","On-Base Percentage","Slugging Percentage",
        "On-base Plus Slugging","Runs Batted In","Runs Scored","Walks","Hit By Pitch","Strikeouts","Extra-Base Hits",
        "Doubles","Triples","Home Runs","Total Bases","Stolen Bases","Pitches per Plate Appearance",
        "Walk-to-Strikeout Ratio","Contact Percentage","Quality At-Bats","Quality At-Bat Percentage",
        "Hard-Hit Balls","Hard-Hit Ball Percentage","Line Drive %","Fly Ball %","Ground Ball %",
        "Batting Average on Balls In Play","Avg. w/ RISP","Two-Out RBIs"
    ]
})

PITCHING_KEY = pd.DataFrame({
    "Acronym": [
        "IP","ERA","WHIP","H","R","ER","BB","BB/INN","SO","K-L","HR","S%","FPS%","FPSO%","FPSH%","SM%","<3%",
        "LD%","FB%","GB%","HHB%","WEAK%","BBS","BAA","BABIP","BA/RISP","CS","SB","SB%","FIP"
    ],
    "Meaning": [
        "Innings Pitched","Earned Run Average","Walks + Hits per Inning","Hits Allowed","Runs Allowed","Earned Runs",
        "Walks","Walks per Inning","Strikeouts","Strikeouts Looking","Home Runs Allowed","Strike %","First-Pitch Strike %",
        "% of FPS ABs that end in outs","% of FPS that are hits","Swinging Miss %","% of ABs with ≤3 pitches",
        "Line Drive %","Fly Ball %","Ground Ball %","Hard-Hit Ball %","Weak Contact %","Base on Balls that results in a run",
        "Batting Avg Against","BABIP","Avg. w/ RISP","Caught Stealing","Stolen Bases Allowed","Stolen Base %",
        "Fielding Independent Pitching"
    ]
})

FIELDING_KEY = pd.DataFrame({
    "Acronym": ["TC","A","PO","FPCT","E","DP"],
    "Meaning": ["Total Chances","Assists","Putouts","Fielding Percentage","Errors","Double Plays involvement"]
})

CATCHING_KEY = pd.DataFrame({
    "Acronym": ["INN","PB","SB-ATT","CS","CS%"],
    "Meaning": ["Innings Caught","Passed Balls","Stolen Base Attempts","Caught Stealing","Caught Stealing %"]
})

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove totals rows and clean name columns."""
    if "Last" in df.columns and "First" in df.columns:
        df["Last"] = df["Last"].astype(str).str.strip()
        df["First"] = df["First"].astype(str).str.strip()

        def _norm_missing(s):
            s = s.astype(str).str.strip()
            lower = s.str.lower()
            return s.mask(lower.isin(["", "nan", "none"]))

        df["Last"] = _norm_missing(df["Last"])
        df["First"] = _norm_missing(df["First"])

        totals_idx = df.index[df["Last"].isna() & df["First"].isna()]
        if len(totals_idx) > 0:
            df = df.loc[: totals_idx[0] - 1].reset_index(drop=True)
    return df

def prepare_batting_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare hitting stats dataframe."""
    df = df.copy()
    cols = [
        "Last","First","PA","AB","H","AVG","OBP","SLG","OPS","RBI","R","BB","HBP","SO","XBH","2B","3B","HR",
        "TB","SB","PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB %","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    if "PA" in df.columns:
        df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
        df = df[df["PA"] >= QUAL_MINS["Hitting"]].reset_index(drop=True)
    if {"Last","First"}.issubset(df.columns):
        df = df.sort_values(["Last","First"]).reset_index(drop=True)
    return df

def prepare_pitching_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare pitching stats dataframe."""
    df = df.copy()
    # Extract pitching columns (columns 53-148 from cumulative format)
    if df.shape[1] > 53:
        df = df.iloc[:, [1, 2] + list(range(53, min(148, df.shape[1])))]
        df.columns = [c.replace(".1", "") for c in df.columns]
    
    cols = [
        "Last","First","IP","ERA","WHIP","H","R","ER","BB","BB/INN","SO","K-L","HR",
        "S%","FPS%","FPSO%","FPSH%","SM%","<3%","LD%","FB%","GB%","HHB%","WEAK%","BBS","BAA","BABIP",
        "BA/RISP","CS","SB","SB%","FIP"
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    if "IP" in df.columns:
        df["IP"] = pd.to_numeric(df["IP"], errors="coerce")
        df = df[df["IP"] >= QUAL_MINS["Pitching"]].reset_index(drop=True)
    for col in df.columns:
        if col not in ["Last","First","BABIP","BAA","BA/RISP"] and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(2)
    if {"Last","First"}.issubset(df.columns):
        df = df.sort_values(["Last","First"]).reset_index(drop=True)
    return df

def prepare_fielding_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare fielding stats dataframe."""
    df = df.copy()
    # Extract fielding columns (columns 148+ from cumulative format)
    if df.shape[1] > 148:
        df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
        df.columns = [c.replace(".1", "").replace(".2", "") for c in df.columns]
    
    cols = ["Last","First","TC","A","PO","FPCT","E","DP"]
    df = df[[c for c in cols if c in df.columns]].copy()
    if "TC" in df.columns:
        df["TC"] = pd.to_numeric(df["TC"], errors="coerce")
        df = df[df["TC"] >= QUAL_MINS["Fielding"]].reset_index(drop=True)
    if {"Last","First"}.issubset(df.columns):
        df = df.sort_values(["Last","First"]).reset_index(drop=True)
    for col in df.columns:
        if col not in ["Last","First","FPCT"] and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(0)
    return df

def prepare_catching_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare catching stats dataframe."""
    df = df.copy()
    # Extract catching columns (columns 148+ from cumulative format)
    if df.shape[1] > 148:
        df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
        df.columns = [c.replace(".1", "").replace(".2", "") for c in df.columns]
    
    cols = ["Last","First","INN","PB","SB-ATT","CS","CS%"]
    df = df[[c for c in cols if c in df.columns]].copy()
    if "INN" in df.columns:
        df["INN"] = pd.to_numeric(df["INN"], errors="coerce")
        df = df[df["INN"] >= QUAL_MINS["Catching"]].reset_index(drop=True)
    return df

def parse_series_filename(filename):
    """Parse series filename to extract opponent, season, and year.
    Format: [Opponent]_[Season]_[Year].csv
    Example: Wake_Fall_2025.csv or Cumulative_Fall_2025.csv
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split("_")
    
    if len(parts) >= 3:
        opponent = parts[0]
        season = parts[1]
        year = parts[2]
        is_cumulative = opponent.lower() == "cumulative"
        return {
            "opponent": opponent,
            "season": season,
            "year": year,
            "filename": filename,
            "is_cumulative": is_cumulative
        }
    return None

def list_series_files():
    """List all CSV files and parse their metadata."""
    series_files = []
    for filepath in glob.glob("*.csv"):
        basename = os.path.basename(filepath)
        parsed = parse_series_filename(basename)
        if parsed:
            series_files.append(parsed)
    return series_files

def get_most_recent_cumulative(series_files):
    """Find the most recent cumulative file based on year and season order."""
    cumulative_files = [s for s in series_files if s.get("is_cumulative", False)]
    
    if not cumulative_files:
        return None
    
    # Define season order (Spring comes after Fall in academic year)
    season_order = {"Fall": 1, "Spring": 2, "Summer": 3}
    
    # Sort by year (descending) then by season
    def sort_key(s):
        year_val = int(s["year"]) if s["year"].isdigit() else 0
        season_val = season_order.get(s["season"], 0)
        return (-year_val, -season_val)
    
    cumulative_files.sort(key=sort_key)
    return cumulative_files[0]

def load_csv(filename):
    """Load a CSV file and return cleaned dataframe."""
    try:
        df = pd.read_csv(filename, header=1, dtype=str)
        df = df.applymap(lambda x: x.strip().replace('"', '') if isinstance(x, str) else x)
        df = df.replace({"-": np.nan, "": np.nan, "N/A": np.nan})
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        return clean_df(df)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return pd.DataFrame()

def get_stat_frames(raw_df, stat_types):
    """Extract stat frames for selected stat types."""
    frames = {}
    
    if "Hitting" in stat_types:
        frames["Hitting"] = prepare_batting_stats(raw_df)
    
    if "Pitching" in stat_types:
        frames["Pitching"] = prepare_pitching_stats(raw_df)
    
    if "Fielding" in stat_types:
        frames["Fielding"] = prepare_fielding_stats(raw_df)
    
    if "Catching" in stat_types:
        frames["Catching"] = prepare_catching_stats(raw_df)
    
    return frames

def format_dataframe(df, stat_type):
    """Format dataframe for display."""
    if df is None or df.empty:
        return df
    
    out = df.copy()
    
    # Format percentage columns
    pct_cols = [c for c in out.columns if isinstance(c, str) and c.endswith("%")]
    
    for c in pct_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    
    # Format decimal columns
    if stat_type == "Hitting":
        decimal_cols = ["AVG", "OBP", "SLG", "OPS", "BABIP", "BA/RISP", "PS/PA"]
        for c in decimal_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
                out[c] = out[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    
    if stat_type == "Pitching":
        decimal_cols = ["ERA", "WHIP", "BB/INN", "BAA", "BABIP"]
        for c in decimal_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
                out[c] = out[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    
    if stat_type == "Fielding":
        if "FPCT" in out.columns:
            out["FPCT"] = pd.to_numeric(out["FPCT"], errors="coerce")
            out["FPCT"] = out["FPCT"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    
    return out

def extract_all_players(frames):
    """Extract all unique player last names from all stat frames."""
    names = set()
    for df in frames.values():
        if df is not None and not df.empty and "Last" in df.columns:
            names.update(df["Last"].dropna().astype(str))
    return sorted(names)

def filter_players(df, selected_lastnames):
    """Filter dataframe by selected player last names."""
    if not selected_lastnames or "Last" not in df.columns:
        return df
    return df[df["Last"].isin(selected_lastnames)].copy()

# Streamlit App
st.set_page_config(page_title="EUCB Stats", layout="wide")
st.title("EUCB Baseball Stats")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    # Get all series files
    series_files = list_series_files()
    
    if not series_files:
        st.error("No CSV files found.")
        st.stop()
    
    # Separate cumulative and regular series
    cumulative_files = [s for s in series_files if s.get("is_cumulative", False)]
    regular_series = [s for s in series_files if not s.get("is_cumulative", False)]
    
    # Get most recent cumulative
    most_recent_cumulative = get_most_recent_cumulative(series_files)
    
    # Series type selection
    series_type = st.radio(
        "View Type",
        ["Cumulative Stats", "Individual Series"],
        index=0,
        help="Choose between cumulative season stats or individual series"
    )
    
    if series_type == "Cumulative Stats":
        if cumulative_files:
            # Create display names for cumulative files
            cumulative_options = []
            cumulative_map = {}
            for cf in cumulative_files:
                display = f"{cf['season']} {cf['year']}"
                cumulative_options.append(display)
                cumulative_map[display] = cf
            
            # Sort options by year/season (most recent first)
            season_order = {"Fall": 1, "Spring": 2, "Summer": 3}
            cumulative_options.sort(
                key=lambda x: (-int(cumulative_map[x]['year']), -season_order.get(cumulative_map[x]['season'], 0))
            )
            
            # Default to most recent
            default_display = f"{most_recent_cumulative['season']} {most_recent_cumulative['year']}" if most_recent_cumulative else cumulative_options[0]
            default_index = cumulative_options.index(default_display) if default_display in cumulative_options else 0
            
            selected_cumulative = st.selectbox(
                "Select Cumulative Period",
                cumulative_options,
                index=default_index
            )
            
            selected_file = cumulative_map[selected_cumulative]
        else:
            st.error("No cumulative files found. Please add files named 'Cumulative_[Season]_[Year].csv'")
            st.stop()
    
    else:  # Individual Series
        if regular_series:
            # Extract unique values from regular series only
            years = sorted(set([s["year"] for s in regular_series]), reverse=True)
            all_seasons = sorted(set([s["season"] for s in regular_series]))
            
            # Default to most recent year
            default_year = years[0] if years else "All"
            
            # Year filter
            year_options = ["All"] + years
            year_index = year_options.index(default_year) if default_year in year_options else 0
            selected_year = st.selectbox("Year", year_options, index=year_index)
            
            # Filter by selected year to get available seasons
            year_filtered = regular_series if selected_year == "All" else [s for s in regular_series if s["year"] == selected_year]
            available_seasons = sorted(set([s["season"] for s in year_filtered]))
            
            # Default to most recent season for the selected year
            season_order = {"Fall": 1, "Spring": 2, "Summer": 3}
            if available_seasons and selected_year != "All":
                # For a specific year, default to the latest season available
                available_seasons_sorted = sorted(available_seasons, key=lambda x: -season_order.get(x, 0))
                default_season = available_seasons_sorted[0]
            else:
                default_season = "All"
            
            # Season filter
            season_options = ["All"] + available_seasons
            season_index = season_options.index(default_season) if default_season in season_options else 0
            selected_season = st.selectbox("Season", season_options, index=season_index)
            
            # Filter series based on year and season
            filtered_series = regular_series
            if selected_year != "All":
                filtered_series = [s for s in filtered_series if s["year"] == selected_year]
            if selected_season != "All":
                filtered_series = [s for s in filtered_series if s["season"] == selected_season]
            
            # Opponent filter (alphabetically sorted, default to first)
            opponents = sorted(set([s["opponent"] for s in filtered_series]))
            
            if opponents:
                selected_opponent = st.selectbox("Opponent", opponents, index=0)
                # Find the matching series file
                matching = [s for s in filtered_series if s["opponent"] == selected_opponent]
                if matching:
                    selected_file = matching[0]
                else:
                    st.error("Could not find matching series file.")
                    st.stop()
            else:
                st.info("No series match the selected filters.")
                st.stop()
        else:
            st.error("No individual series files found.")
            st.stop()
    
    st.divider()
    
    # Stat type filter
    stat_types = st.multiselect(
        "Stat Types",
        STAT_TYPES_ALL,
        default=STAT_TYPES_ALL,
        help="Choose which stat categories to display."
    )

# Load data based on selection
if series_type == "Cumulative Stats":
    st.subheader(f"Cumulative Stats - {selected_file['season']} {selected_file['year']}")
    raw_df = load_csv(selected_file['filename'])
    display_name = f"Cumulative ({selected_file['season']} {selected_file['year']})"
else:
    st.subheader(f"{selected_file['opponent']} - {selected_file['season']} {selected_file['year']}")
    raw_df = load_csv(selected_file['filename'])
    display_name = f"{selected_file['opponent']} ({selected_file['season']} {selected_file['year']})"

# Check if data loaded successfully
if raw_df.empty:
    st.error("No data available.")
    st.stop()

# Get stat frames
frames = get_stat_frames(raw_df, stat_types if stat_types else STAT_TYPES_ALL)

# Extract all player names for filtering
all_player_lastnames = extract_all_players(frames)

# Player filter
selected_players = st.multiselect(
    "Filter by player (Last name); leave empty for All",
    options=all_player_lastnames,
    default=[],
)

# Display tabs
if not stat_types:
    st.warning("Please select at least one stat type.")
    st.stop()

tabs = st.tabs(stat_types)

for tab_name, tab in zip(stat_types, tabs):
    with tab:
        df = frames.get(tab_name, pd.DataFrame())
        
        if df.empty:
            st.info(f"No data available for {tab_name}.")
            continue
        
        # Apply player filter
        df_filtered = filter_players(df, selected_players)
        
        if df_filtered.empty:
            st.warning(f"No {tab_name} data matches the selected player(s).")
            continue
        
        # Format for display
        df_display = format_dataframe(df_filtered, tab_name)
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Show acronym key
        if tab_name == "Hitting":
            with st.expander("Hitting Acronym Key", expanded=False):
                st.dataframe(HITTING_KEY, use_container_width=True, hide_index=True)
        elif tab_name == "Pitching":
            with st.expander("Pitching Acronym Key", expanded=False):
                st.dataframe(PITCHING_KEY, use_container_width=True, hide_index=True)
        elif tab_name == "Fielding":
            with st.expander("Fielding Acronym Key", expanded=False):
                st.dataframe(FIELDING_KEY, use_container_width=True, hide_index=True)
        elif tab_name == "Catching":
            with st.expander("Catching Acronym Key", expanded=False):
                st.dataframe(CATCHING_KEY, use_container_width=True, hide_index=True)
