# eucb_stats_app_refactored.py

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Constants and lookup tables
# -------------------------------------------------------------------

STAT_TYPES = ["Hitting", "Pitching", "Fielding", "Catching"]

QUAL_MINS = {
    "Hitting": 1,       # minimum PA
    "Pitching": 0.1,    # minimum IP
    "Fielding": 1,      # minimum TC
    "Catching": 0.1,    # minimum INN
}

CUMULATIVE_FILE = "cumulative.csv"

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
        "Hard-Hit Balls","Hard-Hit Ball Percentage","Line Drive Percentage","Fly Ball Percentage",
        "Ground Ball Percentage","Batting Average on Balls In Play","Average with RISP","Two-Out RBIs"
    ]
})

PITCHING_KEY = pd.DataFrame({
    "Acronym": [
        "IP","ERA","WHIP","H","R","ER","BB","BB/INN","SO","K-L","HR","S%","FPS%","FPSO%","FPSH%",
        "SM%","<3%","LD%","FB%","GB%","HHB%","WEAK%","BBS","BAA","BABIP","BA/RISP","CS","SB","SB%","FIP"
    ],
    "Meaning": [
        "Innings Pitched","Earned Run Average","Walks plus Hits per Inning","Hits Allowed",
        "Runs Allowed","Earned Runs","Walks","Walks per Inning","Strikeouts","Strikeouts Looking",
        "Home Runs Allowed","Strike Percentage","First-Pitch Strike Percentage",
        "Percent of FPS at bats that end in outs","Percent of FPS that are hits",
        "Swinging Miss Percentage","Percent of at bats with three pitches or fewer",
        "Line Drive Percentage","Fly Ball Percentage","Ground Ball Percentage",
        "Hard-Hit Ball Percentage","Weak Contact Percentage",
        "Base on Balls that result in a run","Batting Average Against",
        "Batting Average on Balls In Play","Average with RISP",
        "Caught Stealing","Stolen Bases Allowed","Stolen Base Percentage",
        "Fielding Independent Pitching"
    ]
})

FIELDING_KEY = pd.DataFrame({
    "Acronym": ["TC","A","PO","FPCT","E","DP"],
    "Meaning": [
        "Total Chances","Assists","Putouts","Fielding Percentage","Errors","Double Plays involvement"
    ]
})

CATCHING_KEY = pd.DataFrame({
    "Acronym": ["INN","PB","SB-ATT","CS","CS%"],
    "Meaning": [
        "Innings Caught","Passed Balls","Stolen Base Attempts","Caught Stealing","Caught Stealing Percentage"
    ]
})


# GameChanger exports put pitching and fielding blocks farther to the right
PITCHING_COL_START = 53
FIELDING_COL_START = 148


# -------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------

def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Last and First columns and drop blank total rows."""
    df = df.copy()
    for col in ["Last", "First"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            s = s.mask(s.str.lower().isin(["", "nan", "none"]))
            df[col] = s

    if "Last" in df.columns or "First" in df.columns:
        df = df.dropna(subset=[c for c in ["Last", "First"] if c in df.columns], how="all")
        if {"Last", "First"}.issubset(df.columns):
            df = df.sort_values(["Last", "First"]).reset_index(drop=True)
    return df


def list_series_csvs() -> list[str]:
    """List individual series csv base names, excluding the cumulative file."""
    candidates = []
    for p in glob.glob("*.csv"):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.lower() != os.path.splitext(CUMULATIVE_FILE)[0].lower():
            candidates.append(base)
    return sorted(candidates)


def load_csv(path: str) -> pd.DataFrame:
    """Load a GameChanger style csv using the second row as header."""
    return pd.read_csv(path, header=1)


def load_cumulative_df() -> pd.DataFrame:
    """Read the cumulative csv from current directory or /mnt/data."""
    candidates = [
        CUMULATIVE_FILE,
        os.path.join("/mnt/data", CUMULATIVE_FILE),
    ]
    candidates.extend(
        [p for p in glob.glob("*.csv") + glob.glob("/mnt/data/*.csv")
         if "cumulative" in os.path.basename(p).lower()]
    )

    for path in candidates:
        try:
            if os.path.exists(path):
                df = load_csv(path).copy()
                # Clean text noise like quotes and hyphens used as blanks
                df = df.applymap(
                    lambda x: x.strip().replace('"', "") if isinstance(x, str) else x
                )
                df = df.replace({"-": np.nan, "": np.nan, "N/A": np.nan})
                return clean_names(df)
        except Exception as exc:
            st.warning(f"Could not read {path}: {exc}")

    st.error("No cumulative csv found")
    return pd.DataFrame()


def convert_innings_gc(ip_val):
    """Convert GameChanger style innings (4.1, 4.2) into 4.333, 4.667."""
    try:
        val = float(ip_val)
    except Exception:
        return np.nan
    whole = int(val)
    tenths = round((val - whole) * 10)
    if tenths == 1:
        return whole + 1 / 3
    if tenths == 2:
        return whole + 2 / 3
    return val


def totals_row_label(tab_name: str) -> dict:
    """Return label values for total row."""
    return {"Last": "Team", "First": "Total"}


def apply_qual_minimum(df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    """Filter players that do not meet the simple minimum threshold."""
    df = df.copy()
    if df.empty:
        return df

    min_val = QUAL_MINS.get(stat_type, 0)
    if stat_type == "Hitting" and "PA" in df.columns:
        df = df[df["PA"] >= min_val]
    elif stat_type == "Pitching" and "IP" in df.columns:
        df = df[df["IP"] >= min_val]
    elif stat_type == "Fielding" and "TC" in df.columns:
        df = df[df["TC"] >= min_val]
    elif stat_type == "Catching" and "INN" in df.columns:
        df = df[df["INN"] >= min_val]
    return df.reset_index(drop=True)


def extract_all_last_names(frames: dict[str, pd.DataFrame]) -> list[str]:
    names = set()
    for df in frames.values():
        if df is not None and not df.empty and "Last" in df.columns:
            names.update(df["Last"].dropna().astype(str))
    return sorted(names)


def filter_players(df: pd.DataFrame, selected_last_names: list[str]) -> pd.DataFrame:
    if not selected_last_names or "Last" not in df.columns:
        return df
    return df[df["Last"].isin(selected_last_names)].copy()


def bold_totals_styler(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Highlight the totals row in bold for better readability."""
    def highlight(row):
        last = str(row.get("Last", "")).strip().lower()
        first = str(row.get("First", "")).strip().lower()
        if last == "team" or last == "totals" or (last == "total" and first == "team"):
            return ["font-weight: bold"] * len(row)
        if first == "total":
            return ["font-weight: bold"] * len(row)
        return [""] * len(row)

    return df.style.apply(highlight, axis=1)


# -------------------------------------------------------------------
# Hitting calculations
# -------------------------------------------------------------------

def build_hitting_from_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw GameChanger hitting table into clean per player stats."""
    cols_needed = [
        "Last","First","PA","AB","H","BB","HBP","SF","TB","R","RBI","SO",
        "2B","3B","HR","SB","CS","QAB","HHB",
        "LD%","FB%","GB%","H_RISP","AB_RISP","PS","2OUTRBI","XBH",
    ]
    df = df_raw[[c for c in cols_needed if c in df_raw.columns]].copy()
    df = clean_names(df)

    numeric_cols = [c for c in df.columns if c not in ["Last", "First"]]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Aggregate if multiple series are concatenated
    df = df.groupby(["Last", "First"], as_index=False).sum()

    # Derived rates
    df["AVG"] = np.where(df["AB"] > 0, df["H"] / df["AB"], 0)
    plate_denom = df["AB"] + df["BB"] + df["HBP"] + df["SF"]
    df["OBP"] = np.where(plate_denom > 0,
                         (df["H"] + df["BB"] + df["HBP"]) / plate_denom, 0)
    df["SLG"] = np.where(df["AB"] > 0, df["TB"] / df["AB"], 0)
    df["OPS"] = df["OBP"] + df["SLG"]
    df["QAB%"] = np.where(df["PA"] > 0, df["QAB"] / df["PA"], 0)
    df["BB/K"] = np.where(df["SO"] > 0, df["BB"] / df["SO"], df["BB"])
    df["C%"] = np.where(df["AB"] > 0, 1 - (df["SO"] / df["AB"]), 0)
    df["HHB%"] = np.where(df["AB"] > 0, df["HHB"] / df["AB"], 0)

    # Rebuild LD / FB / GB percentages from counts if available
    # If only LD% etc are present as rates, leave them as 0 to avoid confusion
    if "LD%" in df_raw.columns and "FB%" in df_raw.columns and "GB%" in df_raw.columns:
        # Interpret LD%, FB%, GB% as fractions if > 1, then convert to counts and reaggregate
        def pct_to_ratio(series):
            values = pd.to_numeric(series, errors="coerce").fillna(0.0)
            return np.where(values > 1.0, values / 100.0, values)

        ratios_ld = pct_to_ratio(df_raw.get("LD%", 0))
        ratios_fb = pct_to_ratio(df_raw.get("FB%", 0))
        ratios_gb = pct_to_ratio(df_raw.get("GB%", 0))

        tmp = pd.DataFrame({
            "Last": df_raw["Last"],
            "First": df_raw["First"],
            "LD": np.rint(ratios_ld * df_raw.get("AB", 0)).astype(int),
            "FB": np.rint(ratios_fb * df_raw.get("AB", 0)).astype(int),
            "GB": np.rint(ratios_gb * df_raw.get("AB", 0)).astype(int),
        })
        tmp = clean_names(tmp)
        tmp = tmp.groupby(["Last", "First"], as_index=False).sum()
        df = df.merge(tmp, on=["Last", "First"], how="left")
    else:
        for c in ["LD", "FB", "GB"]:
            if c not in df.columns:
                df[c] = 0

    total_bip = df["LD"] + df["FB"] + df["GB"]
    df["LD%"] = np.where(total_bip > 0, df["LD"] / total_bip, 0)
    df["FB%"] = np.where(total_bip > 0, df["FB"] / total_bip, 0)
    df["GB%"] = np.where(total_bip > 0, df["GB"] / total_bip, 0)

    denom_babip = df["AB"] - df["SO"] - df["HR"] + df["SF"]
    df["BABIP"] = np.where(
        denom_babip > 0, (df["H"] - df["HR"]) / denom_babip, 0
    )
    df["BA/RISP"] = np.where(df["AB_RISP"] > 0, df["H_RISP"] / df["AB_RISP"], 0)
    df["PS/PA"] = np.where(df["PA"] > 0, df["PS"] / df["PA"], 0)

    float_cols = [
        "AVG","OBP","SLG","OPS","QAB%","BB/K",
        "C%","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","PS/PA"
    ]
    df[float_cols] = df[float_cols].round(3)

    final_cols = [
        "Last","First","PA","AB","AVG","OBP","OPS","SLG","H","R","RBI","BB",
        "2B","3B","HR","SB","QAB","QAB%","BB/K","C%","HHB","HHB%","LD%","FB%",
        "GB%","BABIP","BA/RISP","2OUTRBI","XBH","TB","PS/PA","SO",
    ]
    return df[[c for c in final_cols if c in df.columns]]


def build_hitting_from_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    """Hitting slice for cumulative mode uses the left side of the sheet."""
    return build_hitting_from_frame(raw_all)


def build_hitting_from_series(series_names: list[str]) -> pd.DataFrame:
    """Aggregate hitting stats from one or more series csvs."""
    if not series_names:
        return pd.DataFrame()
    pieces = []
    for name in series_names:
        path = f"{name}.csv"
        if os.path.exists(path):
            pieces.append(load_csv(path))
    if not pieces:
        return pd.DataFrame()
    combined = pd.concat(pieces, ignore_index=True)
    return build_hitting_from_frame(combined)


# -------------------------------------------------------------------
# Pitching calculations
# -------------------------------------------------------------------

def slice_pitching_block(df_raw: pd.DataFrame) -> pd.DataFrame:
    """GameChanger puts pitching starting at a fixed column offset."""
    cols = [1, 2] + list(range(PITCHING_COL_START, FIELDING_COL_START))
    cols = [c for c in cols if c < df_raw.shape[1]]
    tmp = df_raw.iloc[:, cols].copy()
    tmp.columns = [c.replace(".1", "") for c in tmp.columns]
    return clean_names(tmp)


def build_pitching_from_block(df_block: pd.DataFrame) -> pd.DataFrame:
    """Turn a pitching block into per player pitching stats."""
    cols_needed = [
        "Last","First","IP","ER","H","BB","R","SO","K-L","HR",
        "#P","BF","HBP","FPS%","FPSO%","FPSH%","S%","SM%",
        "LD%","FB%","GB%","BABIP","BA/RISP","CS","SB","SB%","<3%","HHB%","WEAK%","BBS",
    ]
    df = df_block[[c for c in cols_needed if c in df_block.columns]].copy()
    df = clean_names(df)

    # Ensure numeric types
    numeric_cols = [c for c in df.columns if c not in ["Last", "First"]]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Convert IP to decimal innings
    if "IP" in df.columns:
        df["IP"] = df["IP"].apply(convert_innings_gc)

    # Derived event counts from percentages
    total_bip = df["BF"] - df["SO"] - df["BB"] - df["HBP"]

    df["Strikes"] = (df["S%"] * df["#P"] / 100).round(0)
    df["FirstPitchStrikes"] = (df["FPS%"] * df["BF"] / 100).round(0)
    df["FPSO"] = (df["FPSO%"] * df["BF"] / 100).round(0)
    df["FPSH"] = (df["FPSH%"] * df["BF"] / 100).round(0)
    df["GroundBalls"] = (df["GB%"] * total_bip / 100).round(0)
    df["FlyBalls"] = (df["FB%"] * total_bip / 100).round(0)
    df["LineDrives"] = (df["LD%"] * total_bip / 100).round(0)
    df["HardHitBalls"] = (df["HHB%"] * total_bip / 100).round(0)
    df["WeakContact"] = (df["WEAK%"] * total_bip / 100).round(0)
    df["Under3Pitches"] = (df["<3%"] * df["BF"] / 100).round(0)
    df["SwingMisses"] = (df["SM%"] * df["#P"] / 100).round(0)

    # Aggregate over series
    df = df.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)

    # Recompute rates from totals
    df["ERA"] = np.where(df["IP"] > 0, (df["ER"] * 9 / df["IP"]), 0)
    df["WHIP"] = np.where(df["IP"] > 0, (df["BB"] + df["H"]) / df["IP"], 0)
    df["BB/INN"] = np.where(df["IP"] > 0, df["BB"] / df["IP"], 0)
    df["FIP"] = np.where(
        df["IP"] > 0, ((13 * df["HR"] + 3 * df["BB"] - 2 * df["SO"]) / df["IP"]) + 3.1, 0
    )
    df["S%"] = np.where(df["#P"] > 0, df["Strikes"] / df["#P"] * 100, 0)
    df["FPS%"] = np.where(df["BF"] > 0, df["FirstPitchStrikes"] / df["BF"] * 100, 0)
    df["FPSO%"] = np.where(df["BF"] > 0, df["FPSO"] / df["BF"] * 100, 0)
    df["FPSH%"] = np.where(df["BF"] > 0, df["FPSH"] / df["BF"] * 100, 0)
    df["SM%"] = np.where(df["#P"] > 0, df["SwingMisses"] / df["#P"] * 100, 0)
    bb_balls = df["BF"] - df["SO"] - df["BB"] - df["HBP"]
    df["LD%"] = np.where(bb_balls > 0, df["LineDrives"] / bb_balls * 100, 0)
    df["FB%"] = np.where(bb_balls > 0, df["FlyBalls"] / bb_balls * 100, 0)
    df["GB%"] = np.where(bb_balls > 0, df["GroundBalls"] / bb_balls * 100, 0)
    df["HHB%"] = np.where(bb_balls > 0, df["HardHitBalls"] / bb_balls * 100, 0)
    df["WEAK%"] = np.where(bb_balls > 0, df["WeakContact"] / bb_balls * 100, 0)
    df["<3%"] = np.where(df["BF"] > 0, df["Under3Pitches"] / df["BF"] * 100, 0)
    df["SB%"] = np.where((df["SB"] + df["CS"]) > 0,
                         df["SB"] / (df["SB"] + df["CS"]) * 100, 0)
    denom_baa = df["BF"] - df["BB"] - df["HBP"]
    df["BAA"] = np.where(denom_baa > 0, df["H"] / denom_baa, 0)
    denom_babip = df["BF"] - df["SO"] - df["HR"] - df["BB"] - df["HBP"]
    df["BABIP"] = np.where(
        denom_babip > 0, (df["H"] - df["HR"]) / denom_babip, 0
    )
    if "BA/RISP" not in df.columns:
        df["BA/RISP"] = 0.0

    float_cols_2 = [
        "ERA","WHIP","BB/INN","FIP","S%","FPS%","FPSO%","FPSH%","SM%",
        "LD%","FB%","GB%","HHB%","WEAK%","<3%","SB%","BAA","BABIP","BA/RISP"
    ]
    df[float_cols_2] = df[float_cols_2].round(2)

    final_cols = [
        "Last","First","IP","ERA","WHIP","SO","K-L","H","R","ER","BB","BB/INN","FIP",
        "S%","FPS%","FPSO%","FPSH%","BAA","BBS","SM%","LD%","FB%","GB%","BABIP",
        "BA/RISP","CS","SB","SB%","<3%","HHB%","WEAK%",
    ]
    return df[[c for c in final_cols if c in df.columns]]


def build_pitching_from_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    block = slice_pitching_block(raw_all)
    return build_pitching_from_block(block)


def build_pitching_from_series(series_names: list[str]) -> pd.DataFrame:
    if not series_names:
        return pd.DataFrame()
    pieces = []
    for name in series_names:
        path = f"{name}.csv"
        if os.path.exists(path):
            pieces.append(load_csv(path))
    if not pieces:
        return pd.DataFrame()
    combined = pd.concat(pieces, ignore_index=True)
    block = slice_pitching_block(combined)
    return build_pitching_from_block(block)


# -------------------------------------------------------------------
# Fielding and catching
# -------------------------------------------------------------------

def slice_fielding_block(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = [1, 2] + list(range(FIELDING_COL_START, df_raw.shape[1]))
    cols = [c for c in cols if c < df_raw.shape[1]]
    tmp = df_raw.iloc[:, cols].copy()
    tmp.columns = [c.replace(".1", "") for c in tmp.columns]
    return clean_names(tmp)


def build_fielding_from_block(df_block: pd.DataFrame) -> pd.DataFrame:
    cols_needed = ["Last","First","TC","A","PO","E","DP"]
    df = df_block[[c for c in cols_needed if c in df_block.columns]].copy()
    df = clean_names(df)

    numeric_cols = [c for c in df.columns if c not in ["Last", "First"]]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df.groupby(["Last", "First"], as_index=False).sum()
    df["FPCT"] = np.where(df["TC"] > 0, (df["A"] + df["PO"]) / df["TC"], 0)
    df["FPCT"] = df["FPCT"].round(3)
    return df[["Last","First","TC","A","PO","FPCT","E","DP"]]


def build_fielding_from_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    return build_fielding_from_block(slice_fielding_block(raw_all))


def build_fielding_from_series(series_names: list[str]) -> pd.DataFrame:
    if not series_names:
        return pd.DataFrame()
    pieces = []
    for name in series_names:
        path = f"{name}.csv"
        if os.path.exists(path):
            pieces.append(load_csv(path))
    if not pieces:
        return pd.DataFrame()
    combined = pd.concat(pieces, ignore_index=True)
    return build_fielding_from_block(slice_fielding_block(combined))


def build_catching_from_block(df_block: pd.DataFrame) -> pd.DataFrame:
    """Catching stats live in the same right side area but use SB-ATT strings."""
    cols_needed = ["Last","First","INN","PB","SB-ATT","CS"]
    df = df_block[[c for c in cols_needed if c in df_block.columns]].copy()
    df = clean_names(df)
    if df.empty:
        return df

    # Parse SB-ATT like "5-8"
    if "SB-ATT" in df.columns:
        split = df["SB-ATT"].astype(str).str.split("-", expand=True)
        if split.shape[1] < 2:
            split[1] = np.nan
        df["SB"] = pd.to_numeric(split[0], errors="coerce").fillna(0)
        df["ATT"] = pd.to_numeric(split[1], errors="coerce").fillna(0)
    else:
        df["SB"] = 0
        df["ATT"] = 0

    df["INN"] = pd.to_numeric(df.get("INN", 0), errors="coerce").fillna(0)
    df["PB"] = pd.to_numeric(df.get("PB", 0), errors="coerce").fillna(0)
    df["CS"] = pd.to_numeric(df.get("CS", 0), errors="coerce").fillna(0)

    df = df.groupby(["Last", "First"], as_index=False).sum()
    df["CS%"] = np.where(df["ATT"] > 0, df["CS"] / df["ATT"] * 100, 0)
    df["CS%"] = df["CS%"].round(1)
    df["SB-ATT"] = df["SB"].astype(int).astype(str) + "-" + df["ATT"].astype(int).astype(str)

    return df[["Last","First","INN","PB","SB-ATT","CS","CS%"]]


def build_catching_from_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    return build_catching_from_block(slice_fielding_block(raw_all))


def build_catching_from_series(series_names: list[str]) -> pd.DataFrame:
    if not series_names:
        return pd.DataFrame()
    pieces = []
    for name in series_names:
        path = f"{name}.csv"
        if os.path.exists(path):
            pieces.append(load_csv(path))
    if not pieces:
        return pd.DataFrame()
    combined = pd.concat(pieces, ignore_index=True)
    return build_catching_from_block(slice_fielding_block(combined))


# -------------------------------------------------------------------
# Totals row building
# -------------------------------------------------------------------

def add_hitting_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = df.copy()

    numeric_cols = [c for c in base.columns if c not in ["Last", "First"]]
    sums = base[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()

    PA = sums.get("PA", 0)
    AB = sums.get("AB", 0)
    H = sums.get("H", 0)
    BB = sums.get("BB", 0)
    HBP = sums.get("HBP", 0)
    SF = sums.get("SF", 0)
    TB = sums.get("TB", 0)
    SO = sums.get("SO", 0)
    HR = sums.get("HR", 0)
    QAB = sums.get("QAB", 0)
    PS = sums.get("PS", 0)
    AB_RISP = sums.get("AB_RISP", 0)
    H_RISP = sums.get("H_RISP", 0)
    HHB = sums.get("HHB", 0)

    row = totals_row_label("Hitting")
    row.update(sums.to_dict())

    row["AVG"] = round(H / AB, 3) if AB else 0
    plate_denom = AB + BB + HBP + SF
    row["OBP"] = round((H + BB + HBP) / plate_denom, 3) if plate_denom else 0
    row["SLG"] = round(TB / AB, 3) if AB else 0
    row["OPS"] = round(row["OBP"] + row["SLG"], 3)
    row["QAB%"] = round(QAB / PA, 3) if PA else 0
    row["BB/K"] = round(BB / SO, 3) if SO else round(BB, 3)
    row["C%"] = round(1 - (SO / AB), 3) if AB else 0
    denom_babip = AB - SO - HR + SF
    row["BABIP"] = round((H - HR) / denom_babip, 3) if denom_babip else 0
    row["BA/RISP"] = round(H_RISP / AB_RISP, 3) if AB_RISP else 0
    row["PS/PA"] = round(PS / PA, 3) if PA else 0
    row["HHB%"] = round(HHB / AB, 3) if AB else 0

    totals_df = pd.DataFrame([row], columns=base.columns)
    return pd.concat([base, totals_df], ignore_index=True)


def add_pitching_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "IP" not in df.columns:
        return df
    base = df.copy()
    numeric_cols = [c for c in base.columns if c not in ["Last", "First"]]
    sums = base[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()

    IP = sums.get("IP", 0.0)
    ER = sums.get("ER", 0.0)
    Hh = sums.get("H", 0.0)
    BBh = sums.get("BB", 0.0)
    HRh = sums.get("HR", 0.0)
    SOh = sums.get("SO", 0.0)
    BF = sums.get("BF", 0.0)
    HBP = sums.get("HBP", 0.0)
    SB = sums.get("SB", 0.0)
    CS = sums.get("CS", 0.0)

    row = totals_row_label("Pitching")
    row.update(sums.to_dict())
    row["ERA"] = round((ER * 9 / IP), 2) if IP else 0
    row["WHIP"] = round((BBh + Hh) / IP, 2) if IP else 0
    row["BB/INN"] = round(BBh / IP, 2) if IP else 0
    row["FIP"] = round(((13 * HRh + 3 * BBh - 2 * SOh) / IP) + 3.1, 2) if IP else 0
    row["SB%"] = round(SB / (SB + CS) * 100, 2) if (SB + CS) else 0
    row["BAA"] = round(Hh / (BF - BBh - HBP), 3) if (BF - BBh - HBP) > 0 else 0
    row["BABIP"] = round((Hh - HRh) / (BF - SOh - HRh - BBh - HBP), 3) \
        if (BF - SOh - HRh - BBh - HBP) > 0 else 0

    totals_df = pd.DataFrame([row], columns=base.columns)
    return pd.concat([base, totals_df], ignore_index=True)


def add_fielding_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = df.copy()
    numeric_cols = [c for c in base.columns if c not in ["Last", "First"]]
    sums = base[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()

    row = totals_row_label("Fielding")
    row.update(sums.to_dict())
    TC = sums.get("TC", 0)
    A = sums.get("A", 0)
    PO = sums.get("PO", 0)
    row["FPCT"] = round((A + PO) / TC, 3) if TC else 0

    totals_df = pd.DataFrame([row], columns=base.columns)
    return pd.concat([base, totals_df], ignore_index=True)


def add_catching_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = df.copy()
    row = totals_row_label("Catching")

    # numeric sums
    tmp = base.copy()
    if "SB-ATT" in tmp.columns:
        split = tmp["SB-ATT"].astype(str).str.split("-", expand=True)
        if split.shape[1] < 2:
            split[1] = np.nan
        tmp["SB"] = pd.to_numeric(split[0], errors="coerce").fillna(0)
        tmp["ATT"] = pd.to_numeric(split[1], errors="coerce").fillna(0)
    else:
        tmp["SB"] = 0
        tmp["ATT"] = 0

    tmp["INN"] = pd.to_numeric(tmp.get("INN", 0), errors="coerce").fillna(0)
    tmp["PB"] = pd.to_numeric(tmp.get("PB", 0), errors="coerce").fillna(0)
    tmp["CS"] = pd.to_numeric(tmp.get("CS", 0), errors="coerce").fillna(0)

    sums = tmp[["INN","PB","SB","ATT","CS"]].sum()
    row["INN"] = sums["INN"]
    row["PB"] = sums["PB"]
    row["SB-ATT"] = f"{int(sums['SB'])}-{int(sums['ATT'])}"
    row["CS"] = sums["CS"]
    row["CS%"] = round(
        (sums["CS"] / sums["ATT"] * 100) if sums["ATT"] > 0 else 0, 1
    )

    totals_df = pd.DataFrame([row], columns=base.columns)
    return pd.concat([base, totals_df], ignore_index=True)


def add_totals(df: pd.DataFrame, tab_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    if tab_name == "Hitting":
        return add_hitting_totals(df)
    if tab_name == "Pitching":
        return add_pitching_totals(df)
    if tab_name == "Fielding":
        return add_fielding_totals(df)
    if tab_name == "Catching":
        return add_catching_totals(df)
    return df


# -------------------------------------------------------------------
# Streamlit app
# -------------------------------------------------------------------

st.set_page_config(page_title="EUCB Stats (Fall 2025)", layout="wide")
st.title("EUCB Stats (Fall 2025)")

with st.sidebar:
    st.header("Filters")

    source_mode = st.radio(
        "Data source",
        ["Cumulative", "Series"],
        index=0,
        help=(
            "Cumulative uses season to date stats from a single cumulative csv. "
            "Series lets you load one or more series csvs and aggregates them."
        ),
    )

    stat_types = st.multiselect(
        "Stat type(s)",
        STAT_TYPES,
        default=STAT_TYPES,
        help="Choose which groups to display.",
    )

    series_options = list_series_csvs()
    selected_series = []
    if source_mode == "Series":
        selected_series = st.multiselect(
            "Series (one or many)",
            options=series_options,
            default=series_options[:1] if series_options else [],
            help="Series correspond to csv base names such as wake, jmu, unc.",
        )

# Build data frames for each stat type
frames: dict[str, pd.DataFrame] = {t: pd.DataFrame() for t in STAT_TYPES}

if source_mode == "Cumulative":
    raw_all = load_cumulative_df()
    if not raw_all.empty:
        if "Hitting" in stat_types:
            frames["Hitting"] = build_hitting_from_cumulative(raw_all)
        if "Pitching" in stat_types:
            frames["Pitching"] = build_pitching_from_cumulative(raw_all)
        if "Fielding" in stat_types:
            frames["Fielding"] = build_fielding_from_cumulative(raw_all)
        if "Catching" in stat_types:
            frames["Catching"] = build_catching_from_cumulative(raw_all)
else:
    if not selected_series:
        st.warning("Select at least one series to view stats.")
        st.stop()
    if "Hitting" in stat_types:
        frames["Hitting"] = build_hitting_from_series(selected_series)
    if "Pitching" in stat_types:
        frames["Pitching"] = build_pitching_from_series(selected_series)
    if "Fielding" in stat_types:
        frames["Fielding"] = build_fielding_from_series(selected_series)
    if "Catching" in stat_types:
        frames["Catching"] = build_catching_from_series(selected_series)

# Apply qualification minimums
for key in STAT_TYPES:
    if key in stat_types:
        frames[key] = apply_qual_minimum(frames[key], key)

all_names = extract_all_last_names({k: v for k, v in frames.items() if k in stat_types})
selected_players = st.multiselect(
    "Filter by player (Last name). Leave empty for all.",
    options=all_names,
    default=[],
)

tabs_to_show = stat_types if stat_types else STAT_TYPES
tabs = st.tabs(tabs_to_show)

for tab_name, tab in zip(tabs_to_show, tabs):
    with tab:
        df = frames.get(tab_name, pd.DataFrame())
        if df.empty:
            st.info(f"No data for {tab_name} with current filters.")
            continue

        df_filtered = filter_players(df, selected_players)
        if tab_name == "Pitching" and "IP" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["IP"] > 0]

        if df_filtered.empty:
            st.warning(f"No {tab_name} rows match the selected filters.")
            continue

        df_with_totals = add_totals(df_filtered, tab_name)

        st.subheader(f"{tab_name} stats")

        styled = bold_totals_styler(df_with_totals)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        if tab_name == "Hitting":
            with st.expander("Hitting acronym key"):
                st.dataframe(HITTING_KEY, use_container_width=True, hide_index=True)
        elif tab_name == "Pitching":
            with st.expander("Pitching acronym key"):
                st.dataframe(PITCHING_KEY, use_container_width=True, hide_index=True)
        elif tab_name == "Fielding":
            with st.expander("Fielding acronym key"):
                st.dataframe(FIELDING_KEY, use_container_width=True, hide_index=True)
        elif tab_name == "Catching":
            with st.expander("Catching acronym key"):
                st.dataframe(CATCHING_KEY, use_container_width=True, hide_index=True)
