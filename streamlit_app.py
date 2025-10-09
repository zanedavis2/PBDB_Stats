import os
import glob
import math
import numpy as np
import pandas as pd
import streamlit as st

# -------------
# UI Constants
# -------------
STAT_TYPES_ALL = ["Hitting", "Pitching", "Fielding", "Catching"]

HITTING_KEY = pd.DataFrame({
    "Acronym": ["PA","AB","H","AVG","OBP","SLG","OPS","RBI","R","BB","SO","XBH","2B","3B","HR","TB","SB",
                "PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI"],
    "Meaning":  ["Plate Appearances","At-Bats","Hits","Batting Average","On-Base Percentage","Slugging Percentage",
                 "On-base Plus Slugging","Runs Batted In","Runs Scored","Walks","Strikeouts","Extra-Base Hits",
                 "Doubles","Triples","Home Runs","Total Bases","Stolen Bases","Pitches per Plate Appearance",
                 "Walk-to-Strikeout Ratio","Contact Percentage","Quality At-Bats","Quality At-Bat Percentage",
                 "Hard-Hit Balls","Hard-Hit Ball Percentage","Line Drive Percentage","Fly Ball Percentage",
                 "Ground Ball Percentage","Batting Average on Balls In Play",
                 "Batting Average with RISP","Two-Out RBIs"]
})

PITCHING_KEY = pd.DataFrame({
    "Acronym": ["IP","ERA","WHIP","H","R","ER","BB","BB/INN","SO","K-L","HR","S%","FPS%","FPSO%","FPSH%","SM%","<3%",
                "LD%","FB%","GB%","HHB%","WEAK%","BBS","BAA","BABIP","BA/RISP","CS","SB","SB%","FIP"],
    "Meaning":  ["Innings Pitched","Earned Run Average","Walks+Hits per Inning","Hits Allowed","Runs Allowed",
                 "Earned Runs","Walks","Walks per Inning","Strikeouts","Strikeouts Looking","Home Runs Allowed",
                 "Strike Percentage","First-Pitch Strike %","% of FPS ABs that end in outs","% of FPS that are hits",
                 "Swinging Miss %","% of ABs with ≤3 pitches","Line Drive %","Fly Ball %","Ground Ball %",
                 "Hard-Hit Ball %","Weak Contact %","Base on Balls that score","Batting Average Against",
                 "Batting Average on Balls In Play","BA with RISP","Caught Stealing","Stolen Bases Allowed",
                 "Stolen Base %","Fielding Independent Pitching"]
})

# ------------------------
# General Helper Functions
# ------------------------
def _to_num(s):
    """Numeric with NaN->0 (works for scalars, Series, or DataFrames)."""
    import pandas as pd
    import numpy as np

    # DataFrame: convert each column
    if isinstance(s, pd.DataFrame):
        return s.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Series or 1-D array-like
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    # Scalars (or anything else): best effort
    try:
        return float(s)
    except Exception:
        return 0.0

def _safe_div(num, den):
    num = _to_num(num)
    den = _to_num(den)
    out = np.where(den != 0, num / den, 0.0)
    return pd.Series(out)

def _convert_innings_value(ip):
    """
    Convert baseball tenths-style innings to decimal innings:
    4.1 -> 4 + 1/3, 4.2 -> 4 + 2/3. Leaves other decimals alone.
    """
    try:
        if pd.isna(ip):
            return np.nan
        ip = float(ip)
        whole = int(ip)
        frac_tenths = round((ip - whole) * 10)
        if frac_tenths == 1:
            return whole + (1.0/3.0)
        elif frac_tenths == 2:
            return whole + (2.0/3.0)
        else:
            return ip
    except Exception:
        return np.nan

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace(".1", "").strip() for c in df.columns]

    # 👇 Keep the first occurrence of any duplicate column name
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    if "Last" not in df.columns:
        df["Last"] = ""
    if "First" not in df.columns:
        df["First"] = ""
    df["Last"] = df["Last"].astype(str).str.strip().str.title()
    df["First"] = df["First"].astype(str).str.strip().str.title()
    return df

# -------------------------
# Hitting Prep & Aggregate
# -------------------------
HIT_INPUT_COLS = [
    "Last","First","PA","AB","H","BB","HBP","SF","TB","R","RBI","SO",
    "2B","3B","HR","SB","CS","QAB","HHB","LD","FB","GB","H_RISP","AB_RISP",
    "PS","2 out RBI","XBH"
]

def prepare_batting_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(df)
    # Keep only existing columns from the list
    keep = [c for c in HIT_INPUT_COLS if c in df.columns] + ["Last","First"]
    df = df[[c for c in dict.fromkeys(keep)]].copy()  # dedupe
    # Cast numerics
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])
    # Derived / display columns
    PA = df.get("PA", 0)
    AB = df.get("AB", 0)
    H  = df.get("H", 0)
    BB = df.get("BB", 0)
    HBP = df.get("HBP", 0)
    SF = df.get("SF", 0)
    TB = df.get("TB", 0)
    SO = df.get("SO", 0)

    df["AVG"]   = _safe_div(H, AB).round(3)
    df["OBP"]   = _safe_div(H + BB + HBP, AB + BB + HBP + SF).round(3)
    df["SLG"]   = _safe_div(TB, AB).round(3)
    df["OPS"]   = (df["OBP"] + df["SLG"]).round(3)
    df["QAB%"]  = _safe_div(df.get("QAB", 0), PA).round(3)
    df["HHB %"] = _safe_div(df.get("HHB", 0), PA).round(3)
    # Batted-ball totals (if not provided)
    batted = _to_num(df.get("LD",0)) + _to_num(df.get("FB",0)) + _to_num(df.get("GB",0))
    df["LD%"] = _safe_div(df.get("LD",0), batted).round(3)
    df["FB%"] = _safe_div(df.get("FB",0), batted).round(3)
    df["GB%"] = _safe_div(df.get("GB",0), batted).round(3)
    # BABIP
    df["BABIP"] = _safe_div(H - _to_num(df.get("HR",0)), AB - SO - _to_num(df.get("HR",0)) + SF).round(3)
    # BA/RISP (if not provided as percent)
    if "BA/RISP" not in df.columns:
        df["BA/RISP"] = _safe_div(_to_num(df.get("H_RISP",0)), _to_num(df.get("AB_RISP",0))).round(3)
    # PS/PA
    if "PS" in df.columns:
        df["PS/PA"] = _safe_div(df["PS"], PA).round(3)
    # BB/K
    df["BB/K"] = _safe_div(BB, SO.replace(0, np.nan)).fillna(0.0).round(3)
    # Contact %
    df["C%"] = _safe_div(AB - SO, AB).round(3)

    # Friendly rename for two-out RBI
    if "2 out RBI" in df.columns:
        df.rename(columns={"2 out RBI": "2OUTRBI"}, inplace=True)

    # Order for display
    display_cols = [
        "Last","First","PA","AB","AVG","OBP","OPS","SLG","H","R","RBI","BB","SO","XBH","2B","3B","HR",
        "TB","SB","PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB %","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI"
    ]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing].copy()

def aggregate_stats_hitting(series_names):
    """
    Read multiple series CSVs and return a concatenated raw hitting frame (not yet derived).
    """
    dfs = []
    for name in series_names:
        file = f"{name}.csv"
        if not os.path.exists(file):
            continue
        df = pd.read_csv(file, header=1)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=HIT_INPUT_COLS)
    combined = pd.concat(dfs, ignore_index=True)
    combined = clean_df(combined)
    # Keep only columns relevant to hitting (if mixed export)
    keep = [c for c in HIT_INPUT_COLS if c in combined.columns]
    return combined[["Last","First"] + keep[2:]].copy()

def generate_aggregated_hitting_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Sum numeric by player then compute derived."""
    df = clean_df(df_raw)
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])
    agg = (
        df.groupby(["Last","First"], as_index=False)
          .sum(numeric_only=True)
    )
    return prepare_batting_stats(agg)

# --------------------------
# Pitching Prep & Aggregate
# --------------------------
PIT_INPUT_COLS = [
    "Last","First","IP","ER","H","R","BB","SO","K-L","HR","#P","BF","HBP",
    "S%","FPS%","FPSO%","FPSH%","SM%","LD%","FB%","GB%","BABIP","BA/RISP",
    "CS","SB","SB%","<3%","HHB%","WEAK%","BBS","ERA","WHIP","FIP","BB/INN","BAA"
]

def prepare_pitching_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(df)
    keep = [c for c in PIT_INPUT_COLS if c in df.columns]
    df = df[[c for c in dict.fromkeys(keep)]].copy()
    # Coerce
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])

    # Real innings as decimal
    if "IP" in df.columns:
        df["_IP_DEC"] = df["IP"].apply(_convert_innings_value)
    else:
        df["_IP_DEC"] = 0.0

    # Derive if missing or to ensure consistency
    if "ERA" not in df.columns or df["ERA"].eq(0).all():
        df["ERA"] = (_safe_div(df.get("ER",0)*9.0, df["_IP_DEC"])).round(3)
    if "WHIP" not in df.columns or df["WHIP"].eq(0).all():
        df["WHIP"] = (_safe_div(df.get("BB",0) + df.get("H",0), df["_IP_DEC"])).round(3)
    if "BB/INN" not in df.columns:
        df["BB/INN"] = (_safe_div(df.get("BB",0), df["_IP_DEC"])).round(3)
    if "FIP" not in df.columns:
        # Simple FIP using constant ~3.1 (adjustable)
        C = 3.1
        num = (13*_to_num(df.get("HR",0)) + 3*(_to_num(df.get("BB",0))+_to_num(df.get("HBP",0))) - 2*_to_num(df.get("SO",0)))
        df["FIP"] = (_safe_div(num, df["_IP_DEC"]) + C).round(3)

    # Display ordering
    display_cols = [
        "Last","First","IP","ERA","WHIP","SO","K-L","H","R","ER","BB","BB/INN",
        "FIP","S%","FPS%","FPSO%","FPSH%","BAA","BBS","SM%","LD%","FB%","GB%","BABIP",
        "BA/RISP","CS","SB","SB%","<3%","HHB%","WEAK%"
    ]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing].copy()

def aggregate_stats_pitching(series_names):
    dfs = []
    for name in series_names:
        file = f"{name}.csv"
        if not os.path.exists(file): 
            continue
        df = pd.read_csv(file, header=1)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=PIT_INPUT_COLS)
    combined = pd.concat(dfs, ignore_index=True)
    return clean_df(combined)

def generate_aggregated_pitching_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(df_raw)
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])
    agg = (
        df.groupby(["Last","First"], as_index=False)
          .sum(numeric_only=True)
    )
    return prepare_pitching_stats(agg)

# ---------------------------
# Fielding Prep & Aggregate
# ---------------------------
FLD_INPUT_COLS = ["Last","First","TC","A","PO","FPCT","E","DP"]

def prepare_fielding_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(df)
    keep = [c for c in FLD_INPUT_COLS if c in df.columns]
    df = df[[c for c in dict.fromkeys(keep)]].copy()
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])
    # If FPCT not present, compute = (PO + A) / TC
    if "FPCT" not in df.columns and {"PO","A","TC"}.issubset(df.columns):
        df["FPCT"] = _safe_div(df["PO"] + df["A"], df["TC"]).round(3)

    display_cols = ["Last","First","TC","PO","A","E","DP","FPCT"]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing].copy()

def aggregate_stats_fielding(series_names):
    dfs = []
    for name in series_names:
        file = f"{name}.csv"
        if not os.path.exists(file): 
            continue
        df = pd.read_csv(file, header=1)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=FLD_INPUT_COLS)
    combined = pd.concat(dfs, ignore_index=True)
    return clean_df(combined)

# ---------------------------
# Catching Prep & Aggregate
# ---------------------------
CAT_INPUT_COLS = ["Last","First","INN","PB","SB-ATT","CS","CS%","SB","ERA"]

def prepare_catching_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(df)
    keep = [c for c in CAT_INPUT_COLS if c in df.columns]
    df = df[[c for c in dict.fromkeys(keep)]].copy()
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])
    # Compute CS% if missing
    if "CS%" not in df.columns and {"SB-ATT","CS"}.issubset(df.columns):
        df["CS%"] = _safe_div(df["CS"], df["SB-ATT"]).round(3)
    display_cols = ["Last","First","INN","PB","SB-ATT","CS","CS%","SB","ERA"]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing].copy()

def aggregate_stats_catching(series_names):
    dfs = []
    for name in series_names:
        file = f"{name}.csv"
        if not os.path.exists(file): 
            continue
        df = pd.read_csv(file, header=1)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=CAT_INPUT_COLS)
    combined = pd.concat(dfs, ignore_index=True)
    return clean_df(combined)

# -----------------------
# Loading / Orchestrators
# -----------------------
@st.cache_data(show_spinner=False)
def list_series_csvs():
    files = [os.path.basename(p) for p in glob.glob("*.csv")]
    series = []
    for f in files:
        name = os.path.splitext(f)[0]
        if name.lower() != "cumulative":
            series.append(name)
    return sorted(series)

@st.cache_data(show_spinner=False)
def load_cumulative():
    """Load cumulative.csv once and split for each stat type via the prep functions."""
    if not os.path.exists("cumulative.csv"):
        return {"Hitting": pd.DataFrame(),"Pitching": pd.DataFrame(),
                "Fielding": pd.DataFrame(),"Catching": pd.DataFrame()}
    raw = pd.read_csv("cumulative.csv", header=1)
    # Prepare per stat
    return {
        "Hitting":  prepare_batting_stats(raw.copy()),
        "Pitching": prepare_pitching_stats(raw.copy()),
        "Fielding": prepare_fielding_stats(raw.copy()),
        "Catching": prepare_catching_stats(raw.copy()),
    }

@st.cache_data(show_spinner=False)
def load_series(stat_types, series_names):
    out = {}
    if "Hitting" in stat_types:
        agg = aggregate_stats_hitting(series_names)
        out["Hitting"] = generate_aggregated_hitting_df(agg)
    if "Pitching" in stat_types:
        agg = aggregate_stats_pitching(series_names)
        out["Pitching"] = generate_aggregated_pitching_df(agg)
    if "Fielding" in stat_types:
        agg = aggregate_stats_fielding(series_names)
        # group/sum by player for fielding
        for c in agg.columns:
            if c not in ("Last","First"):
                agg[c] = _to_num(agg[c])
        agg = (agg.groupby(["Last","First"], as_index=False).sum(numeric_only=True))
        out["Fielding"] = prepare_fielding_stats(agg)
    if "Catching" in stat_types:
        agg = aggregate_stats_catching(series_names)
        for c in agg.columns:
            if c not in ("Last","First"):
                agg[c] = _to_num(agg[c])
        agg = (agg.groupby(["Last","First"], as_index=False).sum(numeric_only=True))
        out["Catching"] = prepare_catching_stats(agg)
    return out

def filter_players(df, selected_players):
    if not selected_players or df.empty:
        return df
    if "Last" not in df.columns:
        return df
    return df[df["Last"].isin(selected_players)].reset_index(drop=True)

def extract_all_players(frames_dict):
    names = set()
    for df in frames_dict.values():
        if isinstance(df, pd.DataFrame) and {"Last","First"}.issubset(df.columns):
            names.update(df["Last"].dropna().astype(str).tolist())
    return sorted(names)

# =========
# UI Layout
# =========
st.set_page_config(page_title="PBDB Team Stats", layout="wide")

st.title("PBDB Team Stats")
st.caption("Default view: **Cumulative** stats for **All Players** across **all four stat types**.")

with st.sidebar:
    st.header("Filters")
    data_source = st.radio(
        "Data source",
        ["Cumulative (default)", "Filter by Series"],
        index=0,
        help="Cumulative shows season-to-date. 'Filter by Series' lets you pick one or more series CSVs."
    )

    stat_types = st.multiselect(
        "Stat type(s)",
        STAT_TYPES_ALL,
        default=STAT_TYPES_ALL,
        help="Choose which player groups to display."
    )

    series_options = list_series_csvs()
    selected_series = []
    if data_source == "Filter by Series":
        selected_series = st.multiselect(
            "Series (choose one or many)",
            options=series_options,
            default=series_options[:1] if series_options else [],
            help="Series correspond to CSV base names (e.g., wake, jmu)."
        )

# ====================
# Data Loading per UI
# ====================
if data_source == "Cumulative (default)":
    frames = load_cumulative()
else:
    if not selected_series:
        st.warning("Select at least one series to view stats.")
        st.stop()
    frames = load_series(stat_types if stat_types else STAT_TYPES_ALL, selected_series)

# Player filter options are built from the currently loaded data (any stat type)
all_player_lastnames = extract_all_players(frames)
selected_players = st.multiselect(
    "Filter by player (Last name); leave empty for All",
    options=all_player_lastnames,
    default=[],
)

# =====================
# Display (Tabbed View)
# =====================
tabs_to_show = stat_types if stat_types else STAT_TYPES_ALL
tabs = st.tabs(tabs_to_show)

for tab_name, tab in zip(tabs_to_show, tabs):
    with tab:
        df = frames.get(tab_name, pd.DataFrame())
        if df.empty:
            st.info(f"No data for **{tab_name}** with current filters.")
            continue

        df_filtered = filter_players(df, selected_players)
        if df_filtered.empty and selected_players:
            st.warning(f"No **{tab_name}** rows match selected player(s).")
            continue

        st.subheader(f"{tab_name} Stats")
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)

        # Acronym keys for Hitting & Pitching
        if tab_name in {"Hitting", "Pitching"}:
            with st.expander(f"{tab_name} Acronym Key", expanded=False):
                st.dataframe(HITTING_KEY if tab_name == "Hitting" else PITCHING_KEY,
                             use_container_width=True, hide_index=True)
import streamlit as st

