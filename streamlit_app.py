# eucb_stats_app_clean.py

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

STAT_TYPES = ["Hitting", "Pitching", "Fielding", "Catching"]

QUAL_MINS = {
    "Hitting": 1.0,      # minimum PA
    "Pitching": 0.1,     # minimum IP
    "Fielding": 1.0,     # minimum TC
    "Catching": 0.1,     # minimum INN
}

CUMULATIVE_FILE = "cumulative.csv"

HITTING_KEY = pd.DataFrame({
    "Acronym": [
        "PA","AB","H","AVG","OBP","SLG","OPS","RBI","R","BB","SO","XBH","2B","3B","HR","TB","SB",
        "PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB %","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI"
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
        "IP","ERA","WHIP","H","R","ER","BB","BB/INN","SO","K-L","HR","S%","FPS%","FPSO%","FPSH%","SM%","<3%",
        "LD%","FB%","GB%","HHB%","WEAK%","BBS","BAA","BABIP","BA/RISP","CS","SB","SB%","FIP"
    ],
    "Meaning": [
        "Innings Pitched","Earned Run Average","Walks plus Hits per Inning","Hits Allowed","Runs Allowed",
        "Earned Runs","Walks","Walks per Inning","Strikeouts","Strikeouts Looking","Home Runs Allowed",
        "Strike Percentage","First-Pitch Strike Percentage","Percent of FPS ABs that end in outs",
        "Percent of FPS that are hits","Swinging Miss Percentage","Percent of ABs with three pitches or fewer",
        "Line Drive Percentage","Fly Ball Percentage","Ground Ball Percentage","Hard-Hit Ball Percentage",
        "Weak Contact Percentage","Base on Balls that results in a run","Batting Average Against",
        "Batting Average on Balls In Play","Average with RISP","Caught Stealing","Stolen Bases Allowed",
        "Stolen Base Percentage","Fielding Independent Pitching"
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

# GameChanger export offsets
PITCHING_COL_START = 53
FIELDING_COL_START = 148


# -------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------

def _standardize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip, title case, and drop rows with empty Last and First."""
    df = df.copy()
    for col in ["Last", "First"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            s = s.mask(s.str.lower().isin(["", "nan", "none"]))
            df[col] = s
    if {"Last", "First"}.issubset(df.columns):
        df = df.dropna(subset=["Last", "First"], how="all")
        df["Last"] = df["Last"].str.title()
        df["First"] = df["First"].str.title()
        df = df.sort_values(["Last", "First"]).reset_index(drop=True)
    return df


def list_series_csvs() -> list:
    """Return base names of all csvs except the cumulative one."""
    names = []
    for p in glob.glob("*.csv"):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.lower() != os.path.splitext(CUMULATIVE_FILE)[0].lower():
            names.append(base)
    return sorted(names)


def _load_gc_csv(path: str) -> pd.DataFrame:
    """Load a GameChanger csv using second row as column names."""
    return pd.read_csv(path, header=1)


def load_cumulative_df() -> pd.DataFrame:
    """Try to load the cumulative file from current dir or /mnt/data."""
    candidates = [
        CUMULATIVE_FILE,
        os.path.join("/mnt/data", CUMULATIVE_FILE),
    ]
    candidates += [
        p for p in glob.glob("*.csv") + glob.glob("/mnt/data/*.csv")
        if "cumulative" in os.path.basename(p).lower()
    ]

    for path in candidates:
        try:
            if os.path.exists(path):
                df = _load_gc_csv(path)
                df = df.applymap(
                    lambda x: x.strip().replace('"', "") if isinstance(x, str) else x
                )
                df = df.replace({"-": np.nan, "": np.nan, "N/A": np.nan})
                return _standardize_names(df)
        except Exception as exc:
            st.warning(f"Failed reading {path}: {exc}")

    st.error("No valid cumulative csv found.")
    return pd.DataFrame()


def _convert_innings(gc_val):
    """Turn GameChanger style innings (4.1, 4.2) into decimal innings."""
    try:
        v = float(gc_val)
    except Exception:
        return np.nan
    whole = int(v)
    frac = round((v - whole) * 10)
    if frac == 1:
        return whole + 1 / 3
    if frac == 2:
        return whole + 2 / 3
    return v


def _apply_qual_min(df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    """Drop players that do not meet simple minimum thresholds."""
    df = df.copy()
    if df.empty:
        return df
    limit = QUAL_MINS.get(stat_type, 0)
    if stat_type == "Hitting" and "PA" in df.columns:
        df = df[df["PA"] >= limit]
    elif stat_type == "Pitching" and "IP" in df.columns:
        df = df[df["IP"] >= limit]
    elif stat_type == "Fielding" and "TC" in df.columns:
        df = df[df["TC"] >= limit]
    elif stat_type == "Catching" and "INN" in df.columns:
        df = df[df["INN"] >= limit]
    return df.reset_index(drop=True)


def _totals_label() -> dict:
    """Base label for totals row."""
    return {"Last": "Team", "First": "Total"}


def _extract_all_last_names(frames: dict) -> list:
    names = set()
    for df in frames.values():
        if df is not None and not df.empty and "Last" in df.columns:
            names.update(df["Last"].dropna().astype(str))
    return sorted(names)


def _filter_players(df: pd.DataFrame, selected_last_names: list) -> pd.DataFrame:
    if not selected_last_names or "Last" not in df.columns:
        return df
    return df[df["Last"].isin(selected_last_names)].copy()


# -------------------------------------------------------------------
# Hitting
# -------------------------------------------------------------------

def aggregate_hitting_from_series(series_names: list) -> pd.DataFrame:
    """Sum up raw hitting counting stats from one or more series."""
    cols = [
        "Last","First","PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","2B","3B","HR","SB","CS",
        "QAB","HHB","LD%","FB%","GB%","H_RISP","AB_RISP","PS","2OUTRBI","XBH",
    ]

    def _pct_to_ratio(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return np.where(s > 1.0, s / 100.0, s)

    dfs = []
    for name in series_names:
        path = f"{name}.csv"
        if not os.path.exists(path):
            continue
        df = _load_gc_csv(path)
        df = df[[c for c in cols if c in df.columns]].copy()
        df = _standardize_names(df)
        for col in df.columns:
            if col not in ["Last", "First"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        ld_ratio = _pct_to_ratio(df.get("LD%", 0))
        gb_ratio = _pct_to_ratio(df.get("GB%", 0))
        fb_ratio = _pct_to_ratio(df.get("FB%", 0))
        df["LD"] = np.rint(ld_ratio * df.get("AB", 0)).astype(int)
        df["GB"] = np.rint(gb_ratio * df.get("AB", 0)).astype(int)
        df["FB"] = np.rint(fb_ratio * df.get("AB", 0)).astype(int)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)
    for c in ["LD","GB","FB"]:
        if c in agg_df.columns:
            agg_df[c] = agg_df[c].astype(int)
    return agg_df


def build_hitting_from_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hitting rates from combined counting stats."""
    cols = [
        "Last","First","PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","2B","3B","HR",
        "SB","CS","QAB","HHB","LD","FB","GB","H_RISP","AB_RISP","PS","2OUTRBI","XBH",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols].copy()

    for c in df.columns:
        if c not in ["Last", "First"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    agg = df.copy()

    agg["AVG"] = np.where(agg["AB"] > 0, agg["H"] / agg["AB"], 0)
    plate_denom = agg["AB"] + agg["BB"] + agg["HBP"] + agg["SF"]
    agg["OBP"] = np.where(
        plate_denom > 0,
        (agg["H"] + agg["BB"] + agg["HBP"]) / plate_denom,
        0,
    )
    agg["SLG"] = np.where(agg["AB"] > 0, agg["TB"] / agg["AB"], 0)
    agg["OPS"] = agg["OBP"] + agg["SLG"]
    agg["QAB%"] = np.where(agg["PA"] > 0, agg["QAB"] / agg["PA"], 0)
    agg["BB/K"] = np.where(agg["SO"] > 0, agg["BB"] / agg["SO"], agg["BB"])
    agg["C%"] = np.where(agg["AB"] > 0, 1 - (agg["SO"] / agg["AB"]), 0)
    agg["HHB%"] = np.where(agg["AB"] > 0, agg["HHB"] / agg["AB"], 0)

    total_batted = agg["LD"] + agg["FB"] + agg["GB"]
    agg["LD%"] = np.where(total_batted > 0, agg["LD"] / total_batted, 0)
    agg["FB%"] = np.where(total_batted > 0, agg["FB"] / total_batted, 0)
    agg["GB%"] = np.where(total_batted > 0, agg["GB"] / total_batted, 0)

    denom_babip = agg["AB"] - agg["SO"] - agg["HR"] + agg["SF"]
    agg["BABIP"] = np.where(denom_babip > 0,
                            (agg["H"] - agg["HR"]) / denom_babip, 0)
    agg["BA/RISP"] = np.where(agg["AB_RISP"] > 0,
                              agg["H_RISP"] / agg["AB_RISP"], 0)
    agg["PS/PA"] = np.where(agg["PA"] > 0, agg["PS"] / agg["PA"], 0)

    float_cols = [
        "AVG","OBP","SLG","OPS","QAB%","BB/K","C%","HHB%","LD%","FB%","GB%",
        "BABIP","BA/RISP","PS/PA",
    ]
    agg[float_cols] = agg[float_cols].round(3)

    final_cols = [
        "Last","First","PA","AB","AVG","OBP","OPS","SLG","H","R","RBI","BB",
        "2B","3B","HR","SB","QAB","QAB%","BB/K","C%","HHB","HHB%","LD%","FB%",
        "GB%","BABIP","BA/RISP","2OUTRBI","XBH","TB","PS/PA","SO",
    ]
    return agg[[c for c in final_cols if c in agg.columns]]


def build_hitting_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    """Hitting for cumulative mode uses raw cumulative table directly."""
    cols = [
        "Last","First","PA","AB","H","AVG","OBP","SLG","OPS","RBI","R","BB","SO","XBH","2B","3B","HR",
        "TB","SB","PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB %","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI",
    ]
    df = raw_all[[c for c in cols if c in raw_all.columns]].copy()
    df = _standardize_names(df)
    if "PA" in df.columns:
        df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
        df = df[df["PA"] != 0].reset_index(drop=True)
    return df


def build_hitting_series(series_names: list) -> pd.DataFrame:
    agg = aggregate_hitting_from_series(series_names)
    if agg.empty:
        return agg
    return build_hitting_from_agg(agg)


# -------------------------------------------------------------------
# Pitching
# -------------------------------------------------------------------

def _slice_pitching_block(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = [1, 2] + list(range(PITCHING_COL_START, FIELDING_COL_START))
    cols = [c for c in cols if c < df_raw.shape[1]]
    df = df_raw.iloc[:, cols].copy()
    df.columns = [c.replace(".1", "") for c in df.columns]
    return _standardize_names(df)


def aggregate_pitching_from_series(series_names: list) -> pd.DataFrame:
    cols_to_keep = [
        "IP","ER","H","BB","R","SO","K-L","HR","#P","BF","HBP","FPS%","FPSO%","FPSW%","FPSH%","S%",
        "SM%","LD%","FB%","GB%","BABIP","BA/RISP","CS","SB","SB%","<3%","HHB%","WEAK%","BBS",
    ]
    dfs = []
    for name in series_names:
        path = f"{name}.csv"
        if not os.path.exists(path):
            continue
        df_raw = _load_gc_csv(path)
        df_block = _slice_pitching_block(df_raw)
        for col in ["Last","First"]:
            if col not in df_block.columns:
                df_block[col] = ""
        df = df_block[[c for c in cols_to_keep + ["Last","First"] if c in df_block.columns]]
        df = _standardize_names(df)
        for col in df.columns:
            if col not in ["Last", "First"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "IP" in df.columns:
            df["IP"] = df["IP"].apply(_convert_innings)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for col in agg_df.columns:
        if col not in ["Last","First"]:
            agg_df[col] = agg_df[col].fillna(0).round(3)
    return agg_df


def build_pitching_from_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needed = [
        "IP","ER","H","BB","R","SO","K-L","HR","#P","BF","HBP","Strikes",
        "FirstPitchStrikes","FPSO","FPSH","GroundBalls","FlyBalls","LineDrives",
        "HardHitBalls","WeakContact","Under3Pitches","SwingMisses","BBS","CS","SB",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = 0

    df["IP"] = df["IP"].replace(0, np.nan)
    df["BF"] = df["BF"].replace(0, np.nan)
    df["#P"] = df["#P"].replace(0, np.nan)

    df["ERA"] = (df["ER"] * 9 / df["IP"]).round(2)
    df["WHIP"] = ((df["BB"] + df["H"]) / df["IP"]).round(2)
    df["BB/INN"] = (df["BB"] / df["IP"]).round(2)
    df["FIP"] = (((13 * df["HR"]) + (3 * df["BB"]) - (2 * df["SO"])) / df["IP"] + 3.1).round(2)

    df["S%"] = (df["Strikes"] / df["#P"] * 100).round(2)
    df["FPS%"] = (df["FirstPitchStrikes"] / df["BF"] * 100).round(2)
    df["FPSO%"] = (df["FPSO"] / df["BF"] * 100).round(2)
    df["FPSH%"] = (df["FPSH"] / df["BF"] * 100).round(2)
    bb_balls = df["BF"] - df["SO"] - df["BB"] - df["HBP"]
    df["SM%"] = (df["SwingMisses"] / df["#P"] * 100).round(2)
    df["LD%"] = (df["LineDrives"] / bb_balls * 100).round(2)
    df["FB%"] = (df["FlyBalls"] / bb_balls * 100).round(2)
    df["GB%"] = (df["GroundBalls"] / bb_balls * 100).round(2)
    df["HHB%"] = (df["HardHitBalls"] / bb_balls * 100).round(2)
    df["WEAK%"] = (df["WeakContact"] / bb_balls * 100).round(2)
    df["<3%"] = (df["Under3Pitches"] / df["BF"] * 100).round(2)

    df["SB%"] = np.where((df["SB"] + df["CS"]) > 0,
                         (df["SB"] / (df["SB"] + df["CS"]) * 100).round(2), 0)
    df["BAA"] = np.where(
        (df["BF"] - df["BB"] - df["HBP"]) > 0,
        (df["H"] / (df["BF"] - df["BB"] - df["HBP"])).round(3),
        0,
    )
    df["BABIP"] = np.where(
        (df["BF"] - df["SO"] - df["HR"] - df["BB"] - df["HBP"]) > 0,
        ((df["H"] - df["HR"]) / (df["BF"] - df["SO"] - df["HR"] - df["BB"] - df["HBP"])).round(3),
        0,
    )
    if "BA/RISP" not in df.columns:
        df["BA/RISP"] = 0.0

    final_cols = [
        "Last","First","IP","ERA","WHIP","SO","K-L","H","R","ER","BB","BB/INN",
        "FIP","S%","FPS%","FPSO%","FPSH%","BAA","BBS","SM%","LD%","FB%","GB%",
        "BABIP","BA/RISP","CS","SB","SB%","<3%","HHB%","WEAK%",
    ]
    for c in final_cols:
        if c not in df.columns:
            df[c] = 0
    return df[final_cols].copy()


def build_pitching_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    block = _slice_pitching_block(raw_all)
    # cumulative already has rates computed, just sanitize
    df = block.copy()
    if "IP" in df.columns:
        df["IP"] = pd.to_numeric(df["IP"], errors="coerce")
        df = df[df["IP"] != 0].reset_index(drop=True)
    return df


def build_pitching_series(series_names: list) -> pd.DataFrame:
    agg = aggregate_pitching_from_series(series_names)
    if agg.empty:
        return agg
    pitch_df = build_pitching_from_agg(agg)
    if "IP" in pitch_df.columns:
        pitch_df = pitch_df[pitch_df["IP"].fillna(0) > 0].reset_index(drop=True)
    return pitch_df


# -------------------------------------------------------------------
# Fielding
# -------------------------------------------------------------------

def _slice_fielding_block(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = [1, 2] + list(range(FIELDING_COL_START, df_raw.shape[1]))
    cols = [c for c in cols if c < df_raw.shape[1]]
    df = df_raw.iloc[:, cols].copy()
    df.columns = [c.replace(".1", "") for c in df.columns]
    return _standardize_names(df)


def aggregate_fielding_from_series(series_names: list) -> pd.DataFrame:
    cols_to_keep = ["TC","A","PO","E","DP"]
    dfs = []
    for name in series_names:
        path = f"{name}.csv"
        if not os.path.exists(path):
            continue
        df_raw = _load_gc_csv(path)
        df_block = _slice_fielding_block(df_raw)
        for col in ["Last","First"]:
            if col not in df_block.columns:
                df_block[col] = ""
        df = df_block[[c for c in cols_to_keep + ["Last","First"] if c in df_block.columns]]
        df = _standardize_names(df)
        for col in df.columns:
            if col not in ["Last","First"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for col in agg_df.columns:
        if col not in ["Last","First"]:
            agg_df[col] = agg_df[col].fillna(0).round(3)
    return agg_df


def build_fielding_from_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["TC","A","PO","E","DP"]:
        if c not in df.columns:
            df[c] = 0
    df["FPCT"] = np.where(df["TC"] > 0, (df["A"] + df["PO"]) / df["TC"], 0)
    df["FPCT"] = df["FPCT"].round(3)
    cols = ["Last","First","TC","A","PO","FPCT","E","DP"]
    return df[[c for c in cols if c in df.columns]]


def build_fielding_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    block = _slice_fielding_block(raw_all)
    return build_fielding_from_agg(block)


def build_fielding_series(series_names: list) -> pd.DataFrame:
    agg = aggregate_fielding_from_series(series_names)
    if agg.empty:
        return agg
    return build_fielding_from_agg(agg)


# -------------------------------------------------------------------
# Catching
# -------------------------------------------------------------------

def aggregate_catching_from_series(series_names: list) -> pd.DataFrame:
    cols_to_keep = ["INN","PB","SB","SB-ATT","CS"]
    dfs = []
    for name in series_names:
        path = f"{name}.csv"
        if not os.path.exists(path):
            continue
        df_raw = _load_gc_csv(path)
        df_block = _slice_fielding_block(df_raw)
        df_block.columns = [c.replace(".2", "") for c in df_block.columns]
        for col in ["Last","First"]:
            if col not in df_block.columns:
                df_block[col] = ""
        df = df_block[[c for c in cols_to_keep + ["Last","First"] if c in df_block.columns]]
        df = _standardize_names(df)
        for col in df.columns:
            if col not in ["Last","First","SB-ATT"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    if "SB-ATT" in combined.columns:
        split = combined["SB-ATT"].astype(str).str.split("-", expand=True)
        if split.shape[1] < 2:
            split[1] = np.nan
        combined["SB"] = pd.to_numeric(split[0], errors="coerce").fillna(0).astype(int)
        combined["ATT"] = pd.to_numeric(split[1], errors="coerce").fillna(0).astype(int)
    else:
        combined["SB"] = 0
        combined["ATT"] = 0

    agg_df = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for col in agg_df.columns:
        if col not in ["Last","First","INN"]:
            agg_df[col] = agg_df[col].fillna(0).round(0)
    return agg_df


def build_catching_from_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["INN","PB","SB","ATT","CS"]:
        if c not in df.columns:
            df[c] = 0
    df["CS%"] = np.where(df["ATT"] > 0, (df["CS"] / df["ATT"] * 100).round(1), 0)
    df["SB-ATT"] = df["SB"].astype(int).astype(str) + "-" + df["ATT"].astype(int).astype(str)
    cols = ["Last","First","INN","PB","SB-ATT","CS","CS%"]
    return df[cols]


def build_catching_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    block = _slice_fielding_block(raw_all)
    # same shape as series for catching
    agg = aggregate_catching_from_series([])  # no series, but we want structure
    # easier path: reuse parsing logic on block directly
    df = block.copy()
    keep = ["Last","First","INN","PB","SB-ATT","CS"]
    df = df[[c for c in keep if c in df.columns]]
    if df.empty:
        return df
    df = _standardize_names(df)
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

    df = df.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    return build_catching_from_agg(df)


def build_catching_series(series_names: list) -> pd.DataFrame:
    agg = aggregate_catching_from_series(series_names)
    if agg.empty:
        return agg
    return build_catching_from_agg(agg)


# -------------------------------------------------------------------
# Totals rows
# -------------------------------------------------------------------

def add_hitting_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = df.copy()
    numeric_cols = [c for c in base.columns if c not in ["Last","First"]]
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

    row = _totals_label()
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
    numeric_cols = [c for c in base.columns if c not in ["Last","First"]]
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

    row = _totals_label()
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
    numeric_cols = [c for c in base.columns if c not in ["Last","First"]]
    sums = base[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()

    row = _totals_label()
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

    row = _totals_label()
    row["INN"] = sums["INN"]
    row["PB"] = sums["PB"]
    row["SB-ATT"] = f"{int(sums['SB'])}-{int(sums['ATT'])}"
    row["CS"] = sums["CS"]
    row["CS%"] = round((sums["CS"] / sums["ATT"] * 100) if sums["ATT"] > 0 else 0, 1)

    totals_df = pd.DataFrame([row], columns=base.columns)
    return pd.concat([base, totals_df], ignore_index=True)


def add_totals_row(df: pd.DataFrame, tab_name: str) -> pd.DataFrame:
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
            "Cumulative shows season to date from a single cumulative csv. "
            "Series lets you pick one or multiple series csvs and aggregates them."
        ),
    )

    stat_types = st.multiselect(
        "Stat type(s)",
        STAT_TYPES,
        default=STAT_TYPES,
        help="Choose which player groups to display.",
    )

    series_options = list_series_csvs()
    selected_series = []
    if source_mode == "Series":
        selected_series = st.multiselect(
            "Series (choose one or many)",
            options=series_options,
            default=series_options[:1] if series_options else [],
            help="Series correspond to csv base names such as wake, jmu, unc.",
        )

# build frames
frames = {t: pd.DataFrame() for t in STAT_TYPES}

if source_mode == "Cumulative":
    raw_all = load_cumulative_df()
    if not raw_all.empty:
        if "Hitting" in stat_types:
            frames["Hitting"] = build_hitting_cumulative(raw_all)
        if "Pitching" in stat_types:
            frames["Pitching"] = build_pitching_cumulative(raw_all)
        if "Fielding" in stat_types:
            frames["Fielding"] = build_fielding_cumulative(raw_all)
        if "Catching" in stat_types:
            frames["Catching"] = build_catching_cumulative(raw_all)
else:
    if not selected_series:
        st.warning("Select at least one series to view stats.")
        st.stop()
    if "Hitting" in stat_types:
        frames["Hitting"] = build_hitting_series(selected_series)
    if "Pitching" in stat_types:
        frames["Pitching"] = build_pitching_series(selected_series)
    if "Fielding" in stat_types:
        frames["Fielding"] = build_fielding_series(selected_series)
    if "Catching" in stat_types:
        frames["Catching"] = build_catching_series(selected_series)

# apply qualification mins
for key in STAT_TYPES:
    if key in stat_types:
        frames[key] = _apply_qual_min(frames[key], key)

all_names = _extract_all_last_names({k: v for k, v in frames.items() if k in stat_types})

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

        df_filtered = _filter_players(df, selected_players)

        if tab_name == "Pitching" and "IP" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["IP"].fillna(0) > 0]

        if df_filtered.empty:
            st.warning(f"No {tab_name} rows match the selected filters.")
            continue

        df_with_totals = add_totals_row(df_filtered, tab_name)

        st.subheader(f"{tab_name} stats")
        st.dataframe(df_with_totals, use_container_width=True, hide_index=True)

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
