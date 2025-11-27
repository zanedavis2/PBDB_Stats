# eucb_stats_app_clean.py

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# 1. Global config
# -------------------------------------------------------------------

STAT_GROUPS = ["Hitting", "Pitching", "Fielding", "Catching"]
CUMULATIVE_FILE = "cumulative.csv"

# minimums for leaderboard style filters
QUAL_MINS = {
    "Hitting": 1,      # PA
    "Pitching": 0.1,   # IP
    "Fielding": 1,     # TC
    "Catching": 0.1,   # INN
}

# -------------------------------------------------------------------
# 2. Acronym keys (UI only, not used in calculations)
# -------------------------------------------------------------------

HITTING_KEY = pd.DataFrame({
    "Acronym": [
        "PA","AB","H","AVG","OBP","SLG","OPS","RBI","R","BB","SO","XBH","2B","3B","HR","TB","SB",
        "PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI"
    ],
    "Meaning": [
        "Plate Appearances","At-Bats","Hits","Batting Average","On-Base Percentage","Slugging Percentage",
        "On-base Plus Slugging","Runs Batted In","Runs Scored","Walks","Strikeouts","Extra-Base Hits",
        "Doubles","Triples","Home Runs","Total Bases","Stolen Bases","Pitches per Plate Appearance",
        "Walk-to-Strikeout Ratio","Contact Percentage","Quality At-Bats","Quality At-Bat Percentage",
        "Hard-Hit Balls","Hard-Hit Ball Percentage","Line Drive %","Fly Ball %","Ground Ball %",
        "Batting Average on Balls In Play","Avg w/ RISP","Two-Out RBIs"
    ]
})

PITCHING_KEY = pd.DataFrame({
    "Acronym": [
        "IP","ERA","WHIP","H","R","ER","BB","BB/INN","SO","K-L","HR","S%","FPS%","FPSO%","FPSH%","SM%","<3%",
        "LD%","FB%","GB%","HHB%","WEAK%","BBS","BAA","BABIP","BA/RISP","CS","SB","SB%","FIP"
    ],
    "Meaning": [
        "Innings Pitched","Earned Run Average","Walks + Hits per Inning","Hits Allowed","Runs Allowed",
        "Earned Runs","Walks","Walks per Inning","Strikeouts","Strikeouts Looking","Home Runs Allowed",
        "Strike %","First-Pitch Strike %","% of FPS ABs that end in outs","% of FPS that are hits",
        "Swinging Miss %","% of ABs with ≤3 pitches","Line Drive %","Fly Ball %","Ground Ball %",
        "Hard-Hit Ball %","Weak Contact %","Base on Balls that results in a run","Batting Avg Against",
        "BABIP","Avg w/ RISP","Caught Stealing","Stolen Bases Allowed","Stolen Base %","Fielding Independent Pitching"
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

KEY_TABLES = {
    "Hitting": HITTING_KEY,
    "Pitching": PITCHING_KEY,
    "Fielding": FIELDING_KEY,
    "Catching": CATCHING_KEY,
}

# -------------------------------------------------------------------
# 3. Low level helpers
# -------------------------------------------------------------------

def ratio(num, den):
    """Safe division. Returns 0 when denominator is not positive."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.where(den > 0, num / den, 0.0)


def standardize_names(df):
    """Trim and title-case Last/First, drop rows where both are missing."""
    out = df.copy()
    for col in ["Last", "First"]:
        if col in out.columns:
            s = out[col].astype(str).str.strip()
            s = s.mask(s.str.lower().isin(["", "nan", "none"]))
            out[col] = s.str.title()
    if {"Last", "First"}.issubset(out.columns):
        out = out.dropna(subset=["Last", "First"], how="all")
        out = out.sort_values(["Last", "First"])
    return out.reset_index(drop=True)


def parse_ip(value):
    """
    GameChanger IP often uses 4.1 and 4.2 for thirds.
    Convert 4.1 to 4 + 1/3, 4.2 to 4 + 2/3, otherwise return float.
    """
    if pd.isna(value):
        return np.nan
    try:
        v = float(value)
    except Exception:
        return np.nan
    whole = int(v)
    frac = round((v - whole) * 10)
    if frac == 1:
        return whole + 1 / 3
    if frac == 2:
        return whole + 2 / 3
    return v


def list_series_files():
    """Find all CSVs except the cumulative file and return base names."""
    files = []
    for p in glob.glob("*.csv"):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.lower() != os.path.splitext(CUMULATIVE_FILE)[0].lower():
            files.append(base)
    return sorted(files)


def read_gc_csv(path):
    """
    Read a GameChanger style CSV.

    The files usually have a one-line header before the real header,
    so we use header=1 and then clean cells a bit.
    """
    df = pd.read_csv(path, header=1)
    df = df.applymap(lambda x: x.strip().replace('"', '') if isinstance(x, str) else x)
    df = df.replace({"-": np.nan, "": np.nan, "N/A": np.nan})
    return df


def load_cumulative():
    """Read the cumulative.csv file if present, otherwise empty DataFrame."""
    candidates = [CUMULATIVE_FILE] + [
        p for p in glob.glob("*.csv") if "cumulative" in os.path.basename(p).lower()
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            df = read_gc_csv(path)
            return standardize_names(df)
        except Exception as exc:
            st.warning(f"Error reading {path}: {exc}")
    return pd.DataFrame()

# -------------------------------------------------------------------
# 4. Hitting builders
# -------------------------------------------------------------------

HIT_COUNT_COLS = [
    "PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","2B","3B","HR","SB","CS",
    "QAB","HHB","H_RISP","AB_RISP","PS","2OUTRBI","XBH",
]

def compute_hitting_rates(df):
    """Compute rate stats for already aggregated hitting counts."""
    out = df.copy()
    out["AVG"] = ratio(out["H"], out["AB"])
    out["OBP"] = ratio(out["H"] + out["BB"] + out["HBP"],
                       out["AB"] + out["BB"] + out["HBP"] + out["SF"])
    out["SLG"] = ratio(out["TB"], out["AB"])
    out["OPS"] = out["OBP"] + out["SLG"]
    out["QAB%"] = ratio(out["QAB"], out["PA"])
    out["BB/K"] = np.where(out["SO"] > 0, ratio(out["BB"], out["SO"]), out["BB"])
    out["C%"] = 1.0 - ratio(out["SO"], out["AB"])
    out["HHB%"] = ratio(out["HHB"], out["AB"])
    total_batted = out.get("LD", 0) + out.get("FB", 0) + out.get("GB", 0)
    out["LD%"] = ratio(out.get("LD", 0), total_batted)
    out["FB%"] = ratio(out.get("FB", 0), total_batted)
    out["GB%"] = ratio(out.get("GB", 0), total_batted)
    denom = out["AB"] - out["SO"] - out["HR"] + out["SF"]
    out["BABIP"] = ratio(out["H"] - out["HR"], denom)
    out["BA/RISP"] = ratio(out["H_RISP"], out["AB_RISP"])
    out["PS/PA"] = ratio(out["PS"], out["PA"])
    return out


def build_hitting_from_cumulative(raw_all):
    """
    Hitting view from cumulative file.

    We pull relevant columns, group by player, sum counting stats,
    then compute rate metrics.
    """
    if raw_all.empty:
        return pd.DataFrame()
    cols = ["Last", "First"] + [c for c in HIT_COUNT_COLS if c in raw_all.columns]
    df = raw_all[cols].copy()
    for c in df.columns:
        if c not in ["Last", "First"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = standardize_names(df)
    agg = df.groupby(["Last", "First"], as_index=False).sum()
    # try to recover LD/FB/GB as counts if percent columns exist
    for kind in ["LD", "FB", "GB"]:
        pct_col = f"{kind}%"
        if pct_col in raw_all.columns and "AB" in agg.columns:
            pct = pd.to_numeric(raw_all[pct_col], errors="coerce").fillna(0) / 100.0
            temp = raw_all[["Last", "First"]].copy()
            temp[kind] = np.rint(pct * pd.to_numeric(raw_all.get("AB", 0), errors="coerce")).fillna(0)
            temp = standardize_names(temp)
            agg2 = temp.groupby(["Last", "First"], as_index=False).sum()
            agg = agg.merge(agg2, on=["Last", "First"], how="left")
    agg = agg.fillna(0)
    agg = compute_hitting_rates(agg)
    return agg


def build_hitting_from_series(series_names):
    """Aggregate hitting across one or more series files."""
    if not series_names:
        return pd.DataFrame()
    frames = []
    for base in series_names:
        path = f"{base}.csv"
        if not os.path.exists(path):
            continue
        raw = read_gc_csv(path)
        cols = ["Last", "First"] + [c for c in HIT_COUNT_COLS if c in raw.columns]
        df = raw[cols].copy()
        for c in df.columns:
            if c not in ["Last", "First"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        # reconstruct LD/FB/GB counts from percent columns when available
        for kind in ["LD", "FB", "GB"]:
            pct_col = f"{kind}%"
            if pct_col in raw.columns and "AB" in raw.columns:
                pct = pd.to_numeric(raw[pct_col], errors="coerce").fillna(0) / 100.0
                df[kind] = np.rint(pct * pd.to_numeric(raw["AB"], errors="coerce")).fillna(0)
        df = standardize_names(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    agg = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)
    agg = compute_hitting_rates(agg)
    return agg

# -------------------------------------------------------------------
# 5. Pitching builders
# -------------------------------------------------------------------

PITCH_COUNT_COLS = [
    "IP","ER","H","BB","R","SO","K-L","HR","#P","BF","HBP",
    "BBS","CS","SB",  # BBS = walks that score
]

def compute_pitching_rates(df):
    """Compute pitching rate and percent stats from aggregated counts."""
    out = df.copy()
    out["IP"] = out["IP"].apply(parse_ip)
    out["ERA"] = ratio(out["ER"] * 9.0, out["IP"])
    out["WHIP"] = ratio(out["BB"] + out["H"], out["IP"])
    out["BB/INN"] = ratio(out["BB"], out["IP"])
    out["FIP"] = ratio(13 * out["HR"] + 3 * out["BB"] - 2 * out["SO"], out["IP"]) + 3.1
    out["SB%"] = ratio(out["SB"], out["SB"] + out["CS"]) * 100.0
    out["BAA"] = ratio(out["H"], out["BF"] - out["BB"] - out["HBP"])
    # Batted balls for BABIP
    denom_babip = out["BF"] - out["SO"] - out["HR"] - out["BB"] - out["HBP"]
    out["BABIP"] = ratio(out["H"] - out["HR"], denom_babip)

    # derive a few counts used to compute percents
    out["Strikes"] = ratio(out.get("S%", 0) * out["#P"], 100.0) if "S%" in out.columns else 0
    out["FirstPitchStrikes"] = ratio(out.get("FPS%", 0) * out["BF"], 100.0) if "FPS%" in out.columns else 0
    out["FPSO"] = ratio(out.get("FPSO%", 0) * out["BF"], 100.0) if "FPSO%" in out.columns else 0
    out["FPSH"] = ratio(out.get("FPSH%", 0) * out["BF"], 100.0) if "FPSH%" in out.columns else 0

    # full recalculation of percent columns from internal tallies when possible
    if "#P" in out.columns:
        out["S%"] = ratio(out["Strikes"], out["#P"]) * 100.0
        out["SM%"] = ratio(out.get("SwingMisses", 0), out["#P"]) * 100.0
    if "BF" in out.columns:
        out["FPS%"] = ratio(out["FirstPitchStrikes"], out["BF"]) * 100.0
        out["FPSO%"] = ratio(out["FPSO"], out["BF"]) * 100.0
        out["FPSH%"] = ratio(out["FPSH"], out["BF"]) * 100.0
        out["<3%"] = ratio(out.get("Under3Pitches", 0), out["BF"]) * 100.0
        bb_balls = out["BF"] - out["SO"] - out["BB"] - out["HBP"]
        out["LD%"] = ratio(out.get("LineDrives", 0), bb_balls) * 100.0
        out["FB%"] = ratio(out.get("FlyBalls", 0), bb_balls) * 100.0
        out["GB%"] = ratio(out.get("GroundBalls", 0), bb_balls) * 100.0
        out["HHB%"] = ratio(out.get("HardHitBalls", 0), bb_balls) * 100.0
        out["WEAK%"] = ratio(out.get("WeakContact", 0), bb_balls) * 100.0

    return out


def _pitching_from_any(raw_frames):
    """Shared logic: given list of raw pitching frames, aggregate and compute rates."""
    if not raw_frames:
        return pd.DataFrame()
    combined = pd.concat(raw_frames, ignore_index=True)
    for col in combined.columns:
        if col not in ["Last", "First"]:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)
    combined = standardize_names(combined)
    agg = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)
    agg = compute_pitching_rates(agg)
    # filter pitchers who have not thrown
    if "IP" in agg.columns:
        agg = agg[agg["IP"].fillna(0) > 0].reset_index(drop=True)
    return agg


def build_pitching_from_cumulative(raw_all):
    if raw_all.empty:
        return pd.DataFrame()
    cols = ["Last", "First"] + [c for c in PITCH_COUNT_COLS if c in raw_all.columns]
    extra_cols = []
    for suffix in ["%", "SM%", "<3%"]:
        extra_cols.extend([c for c in raw_all.columns if c.endswith(suffix)])
    cols += extra_cols
    cols = list(dict.fromkeys(cols))  # unique, keep order
    df = raw_all[cols].copy()
    return _pitching_from_any([df])


def build_pitching_from_series(series_names):
    if not series_names:
        return pd.DataFrame()
    frames = []
    for base in series_names:
        path = f"{base}.csv"
        if not os.path.exists(path):
            continue
        raw = read_gc_csv(path)
        cols = ["Last", "First"] + [c for c in PITCH_COUNT_COLS if c in raw.columns]
        extra_cols = []
        for suffix in ["%", "SM%", "<3%"]:
            extra_cols.extend([c for c in raw.columns if c.endswith(suffix)])
        cols += extra_cols
        cols = list(dict.fromkeys(cols))
        df = raw[cols].copy()
        frames.append(df)
    return _pitching_from_any(frames)

# -------------------------------------------------------------------
# 6. Fielding and catching builders
# -------------------------------------------------------------------

FIELD_COUNT_COLS = ["TC", "A", "PO", "E", "DP"]
CATCH_COUNT_COLS = ["INN", "PB", "SB", "CS", "SB-ATT"]

def build_fielding_from_cumulative(raw_all):
    if raw_all.empty:
        return pd.DataFrame()
    cols = ["Last", "First"] + [c for c in FIELD_COUNT_COLS if c in raw_all.columns]
    df = raw_all[cols].copy()
    for c in df.columns:
        if c not in ["Last", "First"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = standardize_names(df)
    agg = df.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)
    agg["FPCT"] = ratio(agg["A"] + agg["PO"], agg["TC"])
    return agg


def build_fielding_from_series(series_names):
    if not series_names:
        return pd.DataFrame()
    frames = []
    for base in series_names:
        path = f"{base}.csv"
        if not os.path.exists(path):
            continue
        raw = read_gc_csv(path)
        # in GC export, fielding stats often live in a block near the end
        # to keep this simple, we just pick the columns we care about
        cols = ["Last", "First"] + [c for c in FIELD_COUNT_COLS if c in raw.columns]
        df = raw[cols].copy()
        for c in df.columns:
            if c not in ["Last", "First"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = standardize_names(combined)
    agg = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)
    agg["FPCT"] = ratio(agg["A"] + agg["PO"], agg["TC"])
    return agg


def build_catching_from_cumulative(raw_all):
    if raw_all.empty:
        return pd.DataFrame()
    df = raw_all.copy()
    cols = ["Last", "First"] + [c for c in CATCH_COUNT_COLS if c in df.columns]
    df = df[cols].copy()
    # SB-ATT may be a string like "3-5"
    if "SB-ATT" in df.columns:
        split = df["SB-ATT"].astype(str).str.split("-", expand=True)
        df["SB"] = pd.to_numeric(split[0], errors="coerce").fillna(0)
        df["ATT"] = pd.to_numeric(split[1], errors="coerce").fillna(0)
    for c in df.columns:
        if c not in ["Last", "First", "SB-ATT"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = standardize_names(df)
    agg = df.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)
    agg["CS%"] = ratio(agg["CS"], agg.get("ATT", 0)) * 100.0
    agg["SB-ATT"] = agg.get("SB", 0).astype(int).astype(str) + "-" + agg.get("ATT", 0).astype(int).astype(str)
    return agg[["Last", "First", "INN", "PB", "SB-ATT", "CS", "CS%"]]


def build_catching_from_series(series_names):
    if not series_names:
        return pd.DataFrame()
    frames = []
    for base in series_names:
        path = f"{base}.csv"
        if not os.path.exists(path):
            continue
        raw = read_gc_csv(path)
        cols = ["Last", "First"] + [c for c in CATCH_COUNT_COLS if c in raw.columns]
        df = raw[cols].copy()
        if "SB-ATT" in df.columns:
            split = df["SB-ATT"].astype(str).str.split("-", expand=True)
            df["SB"] = pd.to_numeric(split[0], errors="coerce").fillna(0)
            df["ATT"] = pd.to_numeric(split[1], errors="coerce").fillna(0)
        for c in df.columns:
            if c not in ["Last", "First", "SB-ATT"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = standardize_names(combined)
    agg = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)
    agg["CS%"] = ratio(agg["CS"], agg.get("ATT", 0)) * 100.0
    agg["SB-ATT"] = agg.get("SB", 0).astype(int).astype(str) + "-" + agg.get("ATT", 0).astype(int).astype(str)
    return agg[["Last", "First", "INN", "PB", "SB-ATT", "CS", "CS%"]]

# -------------------------------------------------------------------
# 7. Frame orchestration and filtering
# -------------------------------------------------------------------

BUILDERS = {
    "Cumulative": {
        "Hitting": build_hitting_from_cumulative,
        "Pitching": build_pitching_from_cumulative,
        "Fielding": build_fielding_from_cumulative,
        "Catching": build_catching_from_cumulative,
    },
    "Series": {
        "Hitting": build_hitting_from_series,
        "Pitching": build_pitching_from_series,
        "Fielding": build_fielding_from_series,
        "Catching": build_catching_from_series,
    },
}

def apply_qual_mins(df, group_name):
    """Drop players who do not meet minimum thresholds."""
    if df is None or df.empty:
        return df
    df = df.copy()
    key = None
    if group_name == "Hitting" and "PA" in df.columns:
        key = "PA"
    elif group_name == "Pitching" and "IP" in df.columns:
        key = "IP"
    elif group_name == "Fielding" and "TC" in df.columns:
        key = "TC"
    elif group_name == "Catching" and "INN" in df.columns:
        key = "INN"
    if key:
        df = df[df[key].fillna(0) >= QUAL_MINS.get(group_name, 0)]
    return df.reset_index(drop=True)


def add_team_row(df, group_name):
    """Append a final Team row with aggregate stats."""
    if df is None or df.empty:
        return df
    base = df.copy()
    num_cols = [c for c in base.columns if c not in ["Last", "First"] and pd.api.types.is_numeric_dtype(base[c])]
    totals = base[num_cols].sum(numeric_only=True)
    team = pd.Series(index=base.columns, dtype=object)
    team["Last"] = "Team"
    team["First"] = "Total"
    for col in num_cols:
        team[col] = totals.get(col, 0)
    team_df = pd.DataFrame([team])
    full = pd.concat([base, team_df], ignore_index=True)
    return full


def build_frames(source_mode, stat_groups, series_selection):
    """Build all requested stat tables for the selected source."""
    frames = {}
    raw_all = load_cumulative() if source_mode == "Cumulative" else None
    for group in stat_groups:
        builder = BUILDERS[source_mode][group]
        if source_mode == "Cumulative":
            df = builder(raw_all)
        else:
            df = builder(series_selection)
        df = apply_qual_mins(df, group)
        df = add_team_row(df, group)
        frames[group] = df
    return frames


def list_all_player_lastnames(frames):
    """Collect unique last names across all stat groups for the filter."""
    names = set()
    for df in frames.values():
        if df is not None and not df.empty and "Last" in df.columns:
            names.update(df["Last"].dropna().astype(str))
    return sorted(names)


def filter_by_players(df, names):
    if not names or df is None or df.empty or "Last" not in df.columns:
        return df
    return df[df["Last"].isin(names)].reset_index(drop=True)

# -------------------------------------------------------------------
# 8. Formatting for display
# -------------------------------------------------------------------

def format_for_display(df, group_name):
    """
    Simple Streamlit friendly formatting:
      * three decimal places for most rates
      * two for pitching ERA/WHIP style metrics
      * percent columns shown with trailing %
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    pct_cols = [c for c in out.columns if isinstance(c, str) and c.endswith("%")]
    for c in pct_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

    # integer like columns
    int_like = {
        "Hitting":  ["PA","AB","H","R","RBI","BB","SO","2B","3B","HR","SB","QAB","XBH","TB","2OUTRBI","H_RISP","AB_RISP","HHB"],
        "Pitching": ["H","R","ER","BB","SO","HR","BBS","CS","SB","BF","#P","HBP"],
        "Fielding": ["TC","A","PO","E","DP"],
        "Catching": ["INN","PB","CS"],
    }.get(group_name, [])

    for c in out.columns:
        if c in ["Last", "First"] or c in pct_cols:
            continue
        if c in int_like:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).round(0).astype(int).astype(str)
        else:
            if pd.api.types.is_numeric_dtype(out[c]):
                # pitching ship: IP, ERA, WHIP, BB/INN to two decimals
                if group_name == "Pitching" and c in ["IP", "ERA", "WHIP", "BB/INN", "FIP", "BAA", "BABIP"]:
                    out[c] = out[c].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")
                else:
                    out[c] = out[c].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
    return out

# -------------------------------------------------------------------
# 9. Streamlit UI
# -------------------------------------------------------------------

def main():
    st.set_page_config(page_title="EUCB Stats (Fall 2025)", layout="wide")
    st.title("EUCB Stats (Fall 2025)")

    with st.sidebar:
        st.header("Filters")
        source_mode = st.radio(
            "Data source",
            ["Cumulative", "Series"],
            index=0,
            help="Cumulative uses season totals from cumulative.csv. Series lets you pick one or more series files and aggregates them."
        )
        stat_groups = st.multiselect(
            "Stat group(s)",
            STAT_GROUPS,
            default=STAT_GROUPS,
            help="Choose which player groups to display."
        )

        series_options = list_series_files()
        if source_mode == "Series":
            if not series_options:
                st.warning("No series CSV files found in this folder.")
            selected_series = st.multiselect(
                "Series (choose one or many)",
                options=series_options,
                default=series_options[:1] if series_options else [],
                help="Series correspond to CSV base names (for example wake, jmu, unc)."
            )
        else:
            selected_series = []

    if source_mode == "Series" and not selected_series:
        if list_series_files():
            st.info("Select at least one series to view stats.")
        else:
            st.info("Drop series CSV files in this directory to use the Series view.")
        return

    # build all requested tables
    groups_to_use = stat_groups if stat_groups else STAT_GROUPS
    frames = build_frames(source_mode, groups_to_use, selected_series)

    # player filter
    all_last_names = list_all_player_lastnames(frames)
    selected_players = st.multiselect(
        "Filter by player (Last name, optional)",
        options=all_last_names,
        default=[],
    )

    # tabs per stat group
    tabs = st.tabs(groups_to_use)
    for group, tab in zip(groups_to_use, tabs):
        with tab:
            df = frames.get(group, pd.DataFrame())
            if df is None or df.empty:
                st.info(f"No data available for {group} with current settings.")
                continue

            df = filter_by_players(df, selected_players)
            if df.empty:
                st.warning(f"No {group} rows match the selected players.")
                continue

            df_display = format_for_display(df, group)

            st.subheader(f"{group} Stats")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            key_table = KEY_TABLES.get(group)
            if key_table is not None and not key_table.empty:
                with st.expander(f"{group} Acronym Key", expanded=False):
                    st.dataframe(key_table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
