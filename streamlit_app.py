import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

# Import your core stat functions and acronym keys from your existing module
# Change "pbdb_stats" to whatever your module is actually called
from pbdb_stats import (
    clean_df,
    prepare_batting_stats,
    prepare_pitching_stats,
    prepare_fielding_stats,
    prepare_catching_stats,
    aggregate_stats_pitching,
    aggregate_stats_hitting,
    aggregate_stats_fielding,
    aggregate_stats_catching,
    generate_aggregated_pitching_df,
    generate_aggregated_hitting_df,
    generate_aggregated_fielding_df,
    generate_aggregated_catching_df,
    filter_by_player,
    HITTING_KEY,
    PITCHING_KEY,
    FIELDING_KEY,
    CATCHING_KEY,
)

CUMULATIVE_FILE = "cumulative.csv"


# List per series CSV files in the working folder
def list_series_csvs():
    names = []
    for p in glob.glob("*.csv"):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.lower() != "cumulative":
            names.append(base)
    return sorted(names)


# Load and clean the cumulative season CSV
@st.cache_data
def load_cumulative_csv():
    if not os.path.exists(CUMULATIVE_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(CUMULATIVE_FILE, header=1, dtype=str)
        df = df.applymap(
            lambda x: x.strip().replace('"', '') if isinstance(x, str) else x
        )
        df = df.replace({"-": np.nan, "": np.nan, "N/A": np.nan})

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        return clean_df(df)
    except Exception as e:
        st.error(f"Error loading cumulative CSV: {e}")
        return pd.DataFrame()


# Aggregate multiple series CSVs and return one combined DataFrame
@st.cache_data
def load_aggregated_from_series(series_names: list) -> pd.DataFrame:
    if not series_names:
        return pd.DataFrame()

    # You already have aggregation helpers per stat type,
    # here we just read one representative CSV then let your aggregators work.
    # The individual stat builders will call the right aggregate function.
    return pd.DataFrame()  # placeholder, not used directly


# Build stat DataFrames for each tab given the chosen data source
def build_stat_frames(source_mode: str, series_names: list):
    if source_mode == "Cumulative":
        base = load_cumulative_csv()

        hitting_df = prepare_batting_stats(base.copy())
        pitching_df = prepare_pitching_stats(base.copy())
        fielding_df = prepare_fielding_stats(base.copy())
        catching_df = prepare_catching_stats(base.copy())

    else:
        # Series mode uses your aggregate plus generate functions
        pitch_raw = aggregate_stats_pitching(series_names)
        hit_raw = aggregate_stats_hitting(series_names)
        field_raw = aggregate_stats_fielding(series_names)
        catch_raw = aggregate_stats_catching(series_names)

        hitting_df = generate_aggregated_hitting_df(hit_raw)
        pitching_df = generate_aggregated_pitching_df(pitch_raw)
        fielding_df = generate_aggregated_fielding_df(field_raw)
        catching_df = generate_aggregated_catching_df(catch_raw)

        # Apply your cleanup functions for consistent formatting
        hitting_df = prepare_batting_stats(hitting_df)
        pitching_df = prepare_pitching_stats(pitching_df)
        fielding_df = prepare_fielding_stats(fielding_df)
        catching_df = prepare_catching_stats(catching_df)

    return {
        "Hitting": hitting_df,
        "Pitching": pitching_df,
        "Fielding": fielding_df,
        "Catching": catching_df,
    }


# Apply optional player filter to all stat frames
def apply_player_filter(frames: dict, selected_players: list) -> dict:
    if not selected_players:
        return frames

    filtered = {}
    for name, df in frames.items():
        try:
            filtered[name] = filter_by_player(selected_players, df)
        except Exception:
            filtered[name] = df
    return filtered


# Render a stats tab with its acronym key
def render_stats_tab(tab_name: str, df_display: pd.DataFrame):
    st.subheader(f"{tab_name} Stats")

    if df_display.empty:
        st.info("No data available for this view.")
        return

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
    )

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


# Main Streamlit app layout and controls
def main():
    st.title("EUCB Baseball Stats")

    # Sidebar controls
    st.sidebar.header("Filters")

    source_mode = st.sidebar.radio(
        "Data source",
        ["Cumulative", "Series"],
        index=0,
    )

    series_names = []
    if source_mode == "Series":
        available_series = list_series_csvs()
        series_names = st.sidebar.multiselect(
            "Select series files",
            options=available_series,
            default=available_series,
        )

    # Build frames using your core logic
    stat_frames = build_stat_frames(source_mode, series_names)

    # Player filter based on last name
    all_players = sorted(
        {row for df in stat_frames.values() for row in df.get("Last", [])}
    )
    selected_players = st.sidebar.multiselect(
        "Filter by player (Last name)", options=all_players
    )
    stat_frames = apply_player_filter(stat_frames, selected_players)

    # Tabs for each stat group
    hitting_tab, pitching_tab, fielding_tab, catching_tab = st.tabs(
        ["Hitting", "Pitching", "Fielding", "Catching"]
    )

    with hitting_tab:
        render_stats_tab("Hitting", stat_frames["Hitting"])

    with pitching_tab:
        render_stats_tab("Pitching", stat_frames["Pitching"])

    with fielding_tab:
        render_stats_tab("Fielding", stat_frames["Fielding"])

    with catching_tab:
        render_stats_tab("Catching", stat_frames["Catching"])


if __name__ == "__main__":
    main()
