# eucb_stats_app.py
import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# CONSTANTS AND ACRONYM KEYS
# -------------------------------------------------------------------
STAT_TYPES_ALL = ["Hitting", "Pitching", "Fielding", "Catching"]
QUAL_MINS = {"Hitting": 1, "Pitching": 0.1, "Fielding": 1, "Catching": 0.1}
CUMULATIVE_FILE = "cumulative.csv"
TOT_ROW_NAMES = {"totals", "total"}

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

INT_LIKE_BY_TAB = {
    "Hitting":  ["PA","AB","H","R","RBI","BB","SO","2B","3B","HR","SB","QAB","XBH","TB","2OUTRBI","H_RISP","AB_RISP","HHB"],
    "Pitching": ["H","R","ER","BB","SO","HR","BBS","CS","SB","K-L","BF","#P","HBP",
                 "GroundBalls","FlyBalls","LineDrives","HardHitBalls","WeakContact","Under3Pitches","SwingMisses"],
    "Fielding": ["TC","A","PO","E","DP"],
    "Catching": ["INN","PB","CS"],
}

# -------------------------------------------------------------------
# SMALL FORMAT HELPERS
# -------------------------------------------------------------------
def _dot3(x: float) -> str:
    """Format a float as .xxx or -.xxx with three decimals."""
    if x is None or (isinstance(x, str) and not str(x).strip()):
        return ""
    try:
        v = float(str(x).replace(",", "").replace("%", ""))
    except Exception:
        return ""
    s = f"{v:.3f}"
    if 0 <= v < 1:
        return "." + s[2:]
    if -1 < v < 0:
        return "-." + s[3:]
    return s

def _int_str(series: pd.Series) -> pd.Series:
    """Convert numeric series to integer-like string with NaN handled."""
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .round(0)
        .fillna(0)
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "")
    )

def _bold_totals(df: pd.DataFrame):
    """Return style function to bold rows where Last is Totals."""
    def _bold(row):
        is_tot = str(row.get("Last", "")).strip().lower() in TOT_ROW_NAMES
        return ["font-weight: bold" if is_tot else "" for _ in row]
    return _bold

# -------------------------------------------------------------------
# DATA CLEANING AND PREP
# -------------------------------------------------------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize names and drop trailing totals section from raw cumulative."""
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
    """Keep core hitting columns, drop zero PA, sort by name."""
    df = df.copy()
    cols = [
        "Last","First","PA","AB","H","AVG","OBP","SLG","OPS","RBI","R","BB","SO","XBH","2B","3B","HR",
        "TB","SB","PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB %","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    if "PA" in df.columns:
        df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
        df = df[df["PA"] != 0].reset_index(drop=True)
    if {"Last","First"}.issubset(df.columns):
        df = df.sort_values(by=["Last","First"]).reset_index(drop=True)
    return df

def prepare_pitching_stats(df: pd.DataFrame, from_cumulative: bool = False) -> pd.DataFrame:
    """Keep core pitching columns, optionally re-slice cumulative layout, sort by name."""
    df = df.copy()
    if from_cumulative:
        df = df.iloc[:, [1, 2] + list(range(53, 148))]
        df.columns = [c.replace(".1", "") for c in df.columns]
    cols = [
        "Last","First","IP","ERA","WHIP","H","R","ER","BB","BB/INN","SO","K-L","HR",
        "S%","FPS%","FPSO%","FPSH%","SM%","<3%","LD%","FB%","GB%","HHB%","WEAK%","BBS","BAA","BABIP",
        "BA/RISP","CS","SB","SB%","FIP"
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    if "IP" in df.columns:
        df["IP"] = pd.to_numeric(df["IP"], errors="coerce")
        df = df[df["IP"] != 0].reset_index(drop=True)
    for col in df.columns:
        if col not in ["Last","First","BABIP","BAA","BA/RISP"] and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(2)
    if {"Last","First"}.issubset(df.columns):
        df = df.sort_values(by=["Last","First"]).reset_index(drop=True)
    return df

def prepare_fielding_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Keep core fielding columns, drop zero chances, round counts, sort."""
    df = df.copy()
    cols = ["Last","First","TC","A","PO","FPCT","E","DP"]
    df = df[[c for c in cols if c in df.columns]].copy()
    if "TC" in df.columns:
        df["TC"] = pd.to_numeric(df["TC"], errors="coerce")
        df = df[df["TC"] != 0].reset_index(drop=True)
    if {"Last","First"}.issubset(df.columns):
        df = df.sort_values(by=["Last","First"]).reset_index(drop=True)
    for col in df.columns:
        if col not in ["Last","First","FPCT"] and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(0)
    return df

def prepare_catching_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Keep core catching columns, drop zero innings."""
    df = df.copy()
    cols = ["Last","First","INN","PB","SB-ATT","CS","CS%"]
    df = df[[c for c in cols if c in df.columns]].copy()
    if "INN" in df.columns:
        df["INN"] = pd.to_numeric(df["INN"], errors="coerce")
        df = df[df["INN"] != 0].reset_index(drop=True)
    return df

# -------------------------------------------------------------------
# SERIES AGGREGATION
# -------------------------------------------------------------------
def aggregate_stats_pitching(csv_files) -> pd.DataFrame:
    """Aggregate pitching counting stats across selected series CSVs."""
    keep = [
        "IP","ER","H","BB","R","SO","K-L","HR","#P","BF","HBP","FPS%","FPSO%","FPSW%","FPSH%","S%","SM%",
        "LD%","FB%","GB%","BABIP","BA/RISP","CS","SB","SB%","<3%","HHB%","WEAK%","BBS",
    ]
    dfs = []
    for name in csv_files:
        df = pd.read_csv(f"{name}.csv", header=1)
        df = df.iloc[:, [1, 2] + list(range(53, 148))]
        df.columns = [c.replace(".1", "") for c in df.columns]
        for col in ["Last","First"]:
            if col not in df.columns:
                df[col] = ""
        df = df[[c for c in keep + ["Last","First"] if c in df.columns]]
        df["Last"] = df["Last"].astype(str).str.strip().str.title()
        df["First"] = df["First"].astype(str).str.strip().str.title()
        for col in df.columns:
            if col not in ["Last","First"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        def convert_innings(ip):
            """Convert GameChanger decimal innings into thirds."""
            try:
                whole = int(ip)
                fraction = round((ip - whole) * 10)
                if fraction == 1: return whole + 1/3
                if fraction == 2: return whole + 2/3
                return float(ip)
            except Exception:
                return float("nan")

        if "IP" in df.columns:
            df["IP"] = df["IP"].apply(convert_innings)

        df["Strikes"]           = (df["S%"]    * df["#P"] / 100).round(0).astype(int)
        df["FirstPitchStrikes"] = (df["FPS%"]  * df["BF"] / 100).round(0).astype(int)
        df["FPSO"]              = (df["FPSO%"] * df["BF"] / 100).round(0).astype(int)
        df["FPSH"]              = (df["FPSH%"] * df["BF"] / 100).round(0).astype(int)

        bip = df["BF"] - df["SO"] - df["BB"] - df["HBP"]
        df["GroundBalls"]  = (df["GB%"]  * bip / 100).round(0).astype(int)
        df["FlyBalls"]     = (df["FB%"]  * bip / 100).round(0).astype(int)
        df["LineDrives"]   = (df["LD%"]  * bip / 100).round(0).astype(int)
        df["HardHitBalls"] = (df["HHB%"] * bip / 100).round(0).astype(int)
        df["WeakContact"]  = (df["WEAK%"]* bip / 100).round(0).astype(int)
        df["Under3Pitches"]= (df["<3%"]  * df["BF"] / 100).round(0).astype(int)
        df["SwingMisses"]  = (df["SM%"]  * df["#P"] / 100).round(0).astype(int)

        df.drop(columns=[c for c in df.columns if c.endswith("%")], inplace=True, errors="ignore")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for col in agg_df.columns:
        if col not in ["Last","First"]:
            agg_df[col] = agg_df[col].fillna(0).round(3)
    return agg_df

def generate_aggregated_pitching_df(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute full pitching line from aggregated series counts."""
    df = df.copy()
    needed = [
        "IP","ER","H","BB","R","SO","K-L","HR","#P","BF","HBP","Strikes","FirstPitchStrikes","FPSO","FPSH",
        "GroundBalls","FlyBalls","LineDrives","HardHitBalls","WeakContact","Under3Pitches","SwingMisses","BBS","CS","SB"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = 0
    df["IP"] = df["IP"].replace(0, np.nan)
    df["BF"] = df["BF"].replace(0, np.nan)
    df["#P"] = df["#P"].replace(0, np.nan)

    df["ERA"]    = (df["ER"] * 9 / df["IP"]).round(2)
    df["WHIP"]   = ((df["BB"] + df["H"]) / df["IP"]).round(2)
    df["BB/INN"] = (df["BB"] / df["IP"]).round(2)
    df["FIP"]    = (((13*df["HR"]) + (3*df["BB"]) - (2*df["SO"])) / df["IP"] + 3.1).round(2)

    df["S%"]     = (df["Strikes"] / df["#P"] * 100).round(2)
    df["FPS%"]   = (df["FirstPitchStrikes"] / df["BF"] * 100).round(2)
    df["FPSO%"]  = (df["FPSO"] / df["BF"] * 100).round(2)
    df["FPSH%"]  = (df["FPSH"] / df["BF"] * 100).round(2)
    bb_balls     = df["BF"] - df["SO"] - df["BB"] - df["HBP"]
    df["SM%"]    = (df["SwingMisses"] / df["#P"] * 100).round(2)
    df["LD%"]    = (df["LineDrives"] / bb_balls * 100).round(2)
    df["FB%"]    = (df["FlyBalls"]  / bb_balls * 100).round(2)
    df["GB%"]    = (df["GroundBalls"]/ bb_balls * 100).round(2)
    df["HHB%"]   = (df["HardHitBalls"]/bb_balls * 100).round(2)
    df["WEAK%"]  = (df["WeakContact"]/bb_balls * 100).round(2)
    df["<3%"]    = (df["Under3Pitches"]/df["BF"] * 100).round(2)

    df["SB%"] = np.where((df["SB"] + df["CS"]) > 0, (df["SB"] / (df["SB"] + df["CS"]) * 100).round(2), 0)
    df["BAA"] = np.where((df["BF"] - df["BB"] - df["HBP"]) > 0, (df["H"] / (df["BF"] - df["BB"] - df["HBP"])).round(3), 0)
    df["BABIP"] = np.where((df["BF"] - df["SO"] - df["HR"] - df["BB"] - df["HBP"]) > 0,
                           ((df["H"] - df["HR"]) / (df["BF"] - df["SO"] - df["HR"] - df["BB"] - df["HBP"])).round(3), 0)
    if "BA/RISP" not in df.columns:
        df["BA/RISP"] = 0.000

    internal = {
        "_IP":"IP","_ER":"ER","_H":"H","_BB":"BB","_HR":"HR","_SO":"SO",
        "_NP":"#P","_BF":"BF","_HBP":"HBP","_STR":"Strikes","_FPS":"FirstPitchStrikes",
        "_FPSO":"FPSO","_FPSH":"FPSH","_GB":"GroundBalls","_FB":"FlyBalls","_LD":"LineDrives",
        "_HHB":"HardHitBalls","_WEAK":"WeakContact","_U3":"Under3Pitches","_SM":"SwingMisses",
        "_BBS":"BBS","_CS":"CS","_SB":"SB"
    }
    for new, old in internal.items():
        df[new] = df[old].fillna(0)

    cols = [
        "Last","First","IP","ERA","WHIP","SO","K-L","H","R","ER","BB","BB/INN","FIP","S%","FPS%","FPSO%","FPSH%",
        "BAA","BBS","SM%","LD%","FB%","GB%","BABIP","BA/RISP","CS","SB","SB%","<3%","HHB%","WEAK%"
    ] + list(internal.keys())
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols].copy()

def aggregate_stats_hitting(csv_files) -> pd.DataFrame:
    """Aggregate hitting counts across series CSVs and derive LD, GB, FB."""
    keep = [
        "Last","First","PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","2B","3B","HR","SB","CS",
        "QAB","HHB","LD%","FB%","GB%","H_RISP","AB_RISP","PS","2OUTRBI","XBH",
    ]

    def _pct_to_ratio(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return np.where(s > 1.0, s / 100.0, s)

    dfs = []
    for name in csv_files:
        df = pd.read_csv(f"{name}.csv", header=1)
        df = df[[c for c in keep if c in df.columns]].copy()
        df["Last"]  = df["Last"].astype(str).str.strip().str.title()
        df["First"] = df["First"].astype(str).str.strip().str.title()
        for col in df.columns:
            if col not in ["Last","First"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        ld_ratio = _pct_to_ratio(df.get("LD%", 0))
        gb_ratio = _pct_to_ratio(df.get("GB%", 0))
        fb_ratio = _pct_to_ratio(df.get("FB%", 0))
        ab = df.get("AB", 0)

        df["LD"] = np.rint(ld_ratio * ab).astype(int)
        df["GB"] = np.rint(gb_ratio * ab).astype(int)
        df["FB"] = np.rint(fb_ratio * ab).astype(int)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for c in ["LD","GB","FB"]:
        if c in agg_df.columns:
            agg_df[c] = agg_df[c].astype(int)
    return agg_df

def generate_aggregated_hitting_df(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute full hitting line from aggregated series counts."""
    cols = [
        "Last","First","PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","2B","3B","HR","SB","CS",
        "QAB","HHB","LD","FB","GB","H_RISP","AB_RISP","PS","2OUTRBI","XBH",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols].copy()
    for c in df.columns:
        if c not in ["Last","First"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    agg = df.groupby(["Last","First"], as_index=False).sum()

    agg["AVG"]   = np.where(agg["AB"] > 0, agg["H"]/agg["AB"], 0)
    agg["OBP"]   = np.where((agg["AB"]+agg["BB"]+agg["HBP"]+agg["SF"]) > 0,
                            (agg["H"]+agg["BB"]+agg["HBP"])/(agg["AB"]+agg["BB"]+agg["HBP"]+agg["SF"]), 0)
    agg["SLG"]   = np.where(agg["AB"] > 0, agg["TB"]/agg["AB"], 0)
    agg["OPS"]   = agg["OBP"] + agg["SLG"]
    agg["QAB%"]  = np.where(agg["PA"] > 0, agg["QAB"]/agg["PA"], 0)
    agg["BB/K"]  = np.where(agg["SO"] > 0, agg["BB"]/agg["SO"], agg["BB"])
    agg["C%"]    = np.where(agg["AB"] > 0, 1 - (agg["SO"]/agg["AB"]), 0)
    agg["HHB%"]  = np.where(agg["AB"] > 0, agg["HHB"]/agg["AB"], 0)

    total_batted = agg["LD"] + agg["FB"] + agg["GB"]
    agg["LD%"]  = np.where(total_batted > 0, agg["LD"]/total_batted, 0)
    agg["FB%"]  = np.where(total_batted > 0, agg["FB"]/total_batted, 0)
    agg["GB%"]  = np.where(total_batted > 0, agg["GB"]/total_batted, 0)

    denom = agg["AB"] - agg["SO"] - agg["HR"] + agg["SF"]
    agg["BABIP"]  = np.where(denom > 0, (agg["H"] - agg["HR"]) / denom, 0)
    agg["BA/RISP"] = np.where(agg["AB_RISP"] > 0, agg["H_RISP"] / agg["AB_RISP"], 0)
    agg["PS/PA"]  = np.where(agg["PA"] > 0, agg["PS"] / agg["PA"], 0)

    pct_cols = ["AVG","OBP","SLG","OPS","QAB%","BB/K","C%","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","PS/PA"]
    agg[pct_cols] = agg[pct_cols].round(3)

    final_cols = [
        "Last","First","PA","AB","AVG","OBP","OPS","SLG","H","R","RBI","BB","2B","3B","HR","SB",
        "QAB","QAB%","BB/K","C%","HHB","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI","XBH","TB","PS/PA","SO",
    ]
    return agg[[c for c in final_cols if c in agg.columns]]

def aggregate_stats_fielding(csv_files) -> pd.DataFrame:
    """Aggregate fielding counts across series CSVs and compute FPCT."""
    keep = ["TC","A","PO","E","DP"]
    dfs = []
    for name in csv_files:
        try:
            df = pd.read_csv(f"{name}.csv", header=1)
            df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
            df.columns = [c.replace(".1", "") for c in df.columns]
            for col in ["Last","First"]:
                if col not in df.columns:
                    df[col] = ""
            df = df[[c for c in keep + ["Last","First"] if c in df.columns]]
            df["Last"]  = df["Last"].astype(str).str.strip().str.title()
            df["First"] = df["First"].astype(str).str.strip().str.title()
            for col in df.columns:
                if col not in ["Last","First"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=keep + ["Last","First"])
    combined = pd.concat(dfs, ignore_index=True)
    agg = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for col in agg.columns:
        if col not in ["Last","First"]:
            agg[col] = agg[col].fillna(0).round(3)
    agg["FPCT"] = ((agg["A"] + agg["PO"]) / agg["TC"]).round(3).fillna(0)
    return agg

def aggregate_stats_catching(csv_files) -> pd.DataFrame:
    """Aggregate catching counts across series CSVs and recompute SB-ATT and CS%."""
    keep = ["INN","PB","SB","SB-ATT","CS"]
    dfs = []
    for name in csv_files:
        try:
            df = pd.read_csv(f"{name}.csv", header=1)
            df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
            df.columns = [c.replace(".1", "").replace(".2", "") for c in df.columns]
            for col in ["Last","First"]:
                if col not in df.columns:
                    df[col] = ""
            df = df[[c for c in keep + ["Last","First"] if c in df.columns]]
            df["Last"]  = df["Last"].astype(str).str.strip().str.title()
            df["First"] = df["First"].astype(str).str.strip().str.title()
            for col in df.columns:
                if col not in ["Last","First","SB-ATT"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=keep + ["Last","First"])

    combined = pd.concat(dfs, ignore_index=True)
    if "SB-ATT" in combined.columns:
        split = combined["SB-ATT"].astype(str).str.split("-", expand=True)
        if split.shape[1] < 2:
            split[1] = np.nan
        combined["SB"]  = pd.to_numeric(split[0], errors="coerce").fillna(0).astype(int)
        combined["ATT"] = pd.to_numeric(split[1], errors="coerce").fillna(0).astype(int)
    else:
        combined["SB"] = 0
        combined["ATT"] = 0

    agg = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for col in agg.columns:
        if col not in ["Last","First","INN"]:
            agg[col] = agg[col].fillna(0).round(0)

    agg["CS%"]    = np.where(agg["ATT"] > 0, (agg["CS"] / agg["ATT"] * 100).round(1), 0)
    agg["SB-ATT"] = agg["SB"].astype(int).astype(str) + "-" + agg["ATT"].astype(int).astype(str)
    return agg.drop(columns=["SB","ATT"])

def _drop_rows_nan_names(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where both Last and First are blank or missing."""
    if df is None or df.empty:
        return df
    for c in [col for col in ["Last","First"] if col in df.columns]:
        s = df[c].astype(str).str.strip()
        df[c] = s.mask(s.str.lower().isin(["","nan","none"]))
    cols = [c for c in ["Last","First"] if c in df.columns]
    if not cols:
        return df
    return df.dropna(subset=cols, how="all").reset_index(drop=True)

def _append_totals(df: pd.DataFrame, tab_name: str) -> pd.DataFrame:
    """Append a Totals row computed from visible rows for a tab."""
    if df is None or df.empty:
        return df
    base = df.copy()
    if "Last" in base.columns:
        lower_last = base["Last"].astype(str).str.strip().str.lower()
        if lower_last.isin(TOT_ROW_NAMES | {""}).any():
            base["_is_total"] = lower_last.isin(TOT_ROW_NAMES)
            base = (pd.concat([base[~base["_is_total"]], base[base["_is_total"]]], ignore_index=True)
                    .drop(columns="_is_total")
                    .reset_index(drop=True))
            return base

    totals = {c: "" for c in base.columns}
    totals["Last"], totals["First"] = "Totals", ""

    def _as_num(s): return pd.to_numeric(s, errors="coerce")
    def ssum(col): return float(_as_num(base[col]).fillna(0).sum()) if col in base.columns else 0.0
    def smean(col):
        if col not in base.columns: return 0.0
        v = _as_num(base[col]).dropna()
        return float(v.mean()) if len(v) else 0.0

    if tab_name == "Hitting":
        PA, AB, H = ssum("PA"), ssum("AB"), ssum("H")
        BB, HBP, SF = ssum("BB"), ssum("HBP"), ssum("SF")
        TB, R, RBI, SO = ssum("TB"), ssum("R"), ssum("RBI"), ssum("SO")
        HR, QAB, PS = ssum("HR"), ssum("QAB"), ssum("PS")
        AB_RISP, H_RISP = ssum("AB_RISP"), ssum("H_RISP")
        for raw in ["PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","HR","QAB","PS","SB","XBH","2B","3B","H_RISP","AB_RISP"]:
            if raw in base.columns:
                totals[raw] = ssum(raw)
        totals["AVG"]     = round(H / AB, 3) if AB else 0
        totals["OBP"]     = round((H + BB + HBP) / (AB + BB + HBP + SF), 3) if (AB + BB + HBP + SF) else 0
        totals["SLG"]     = round(TB / AB, 3) if AB else 0
        totals["OPS"]     = round(totals["OBP"] + totals["SLG"], 3)
        totals["QAB%"]    = round(QAB / PA, 3) if PA else 0
        totals["BB/K"]    = round(BB / SO, 3) if SO else round(BB, 3)
        totals["C%"]      = round(1 - (SO / AB), 3) if AB else 0
        totals["BABIP"]   = round((H - HR) / (AB - SO - HR + SF), 3) if (AB - SO - HR + SF) else 0
        totals["BA/RISP"] = round(H_RISP / AB_RISP, 3) if AB_RISP else 0
        totals["PS/PA"]   = round(PS / PA, 3) if PA else 0
        if "HHB" in base.columns:
            totals["HHB"]  = ssum("HHB")
            totals["HHB%"] = round((totals["HHB"] / AB), 3) if AB else 0
        for c in base.columns:
            if isinstance(c, str) and c.endswith("%") and c not in totals:
                totals[c] = round(smean(c), 3)
        for c in base.columns:
            if c in ["Last","First"] or c in totals: continue
            if pd.api.types.is_numeric_dtype(base[c]): totals[c] = ssum(c)

    elif tab_name == "Pitching":
        for raw in ["IP", "ER", "H", "BB", "HR", "SO", "BF", "HBP", "SB", "CS", "#P"]:
            if raw in base.columns: totals[raw] = ssum(raw)
        series_like = any(col.startswith("_") for col in base.columns)
        if series_like:
            IP  = totals.get("IP", 0.0)
            ER  = totals.get("ER", 0.0)
            Hh  = totals.get("H",  0.0)
            BBh = totals.get("BB", 0.0)
            HRh = totals.get("HR", 0.0)
            SOh = totals.get("SO", 0.0)
            BF  = totals.get("BF", 0.0)
            HBP = totals.get("HBP", 0.0)
            SB  = totals.get("SB", 0.0)
            CS  = totals.get("CS", 0.0)
            totals["ERA"]    = round((ER * 9 / IP), 2) if IP else 0
            totals["WHIP"]   = round((BBh + Hh) / IP, 2) if IP else 0
            totals["BB/INN"] = round(BBh / IP, 2) if IP else 0
            totals["FIP"]    = round(((13 * HRh + 3 * BBh - 2 * SOh) / IP) + 3.1, 2) if IP else 0
            totals["SB%"]    = round((SB / (SB + CS) * 100), 2) if (SB + CS) else 0
            totals["BAA"]    = round(Hh / (BF - BBh - HBP), 3) if (BF - BBh - HBP) > 0 else 0
            totals["BABIP"]  = round((Hh - HRh) / (BF - SOh - HRh - BBh - HBP), 3) if (BF - SOh - HRh - BBh - HBP) > 0 else 0
        else:
            for c in ["ERA", "WHIP", "BB/INN", "FIP"]:
                if c in base.columns:
                    totals[c] = round(smean(c), 2)
            for c in ["BAA", "BABIP"]:
                if c in base.columns:
                    totals[c] = round(smean(c), 3)
            if "SB%" in base.columns:
                totals["SB%"] = round(smean("SB%"), 2)

        pct_cols = [c for c in base.columns if isinstance(c, str) and c.endswith("%")]
        for c in pct_cols:
            col = pd.to_numeric(base[c], errors="coerce")
            totals[c] = round(col.mean(skipna=True), 2) if len(col.dropna()) else 0.0

        for c in base.columns:
            if c in ["Last", "First"] or c in totals:
                continue
            if pd.api.types.is_numeric_dtype(base[c]):
                totals[c] = ssum(c)
            else:
                totals[c] = ""

    elif tab_name == "Fielding":
        for raw in ["TC","A","PO","E","DP"]:
            if raw in base.columns: totals[raw] = ssum(raw)
        TC = totals.get("TC",0); A = totals.get("A",0); PO = totals.get("PO",0)
        totals["FPCT"] = round(((A + PO) / TC), 3) if TC else 0

    elif tab_name == "Catching":
        for raw in ["INN","PB","CS"]:
            if raw in base.columns: totals[raw] = ssum(raw)
        if "SB-ATT" in base.columns:
            split = base["SB-ATT"].astype(str).str.split("-", expand=True)
            sb_sum  = pd.to_numeric(split[0], errors="coerce").fillna(0).sum()
            att_sum = pd.to_numeric(split[1], errors="coerce").fillna(0).sum()
            totals["SB-ATT"] = f"{int(sb_sum)}-{int(att_sum)}"
            totals["CS%"] = round(((att_sum - sb_sum) / att_sum * 100), 1) if att_sum else 0

    for c in base.columns:
        if c in totals or c in ["Last","First"]: continue
        if isinstance(c, str) and c.endswith("%"): totals[c] = round(smean(c), 3)
        elif pd.api.types.is_numeric_dtype(base[c]): totals[c] = ssum(c)
        else: totals[c] = ""

    totals_df = pd.DataFrame([totals]).reindex(columns=base.columns)
    if "Last" in base.columns:
        mask = base["Last"].astype(str).str.strip().str.lower().isin(TOT_ROW_NAMES)
        base = base[~mask]
    return pd.concat([base, totals_df], ignore_index=True)

def _pitching_ip_gt_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Filter pitching rows to IP greater than zero."""
    if "IP" not in df.columns:
        return df
    return df[df["IP"].fillna(0) > 0].copy()

# -------------------------------------------------------------------
# FORMATTERS FOR SERIES AND CUMULATIVE
# -------------------------------------------------------------------
def _format_series(df: pd.DataFrame, tab_name: str):
    """Format series view stats via Styler.format without breaking numeric sort."""
    if df is None or df.empty:
        return df, {}

    out = df.copy()
    pct_cols = [c for c in out.columns if isinstance(c, str) and c.endswith("%")]
    int_like = set(INT_LIKE_BY_TAB.get(tab_name, []))
    format_dict = {}

    # Hitting rate stats as .xxx
    if tab_name == "Hitting":
        for c in ["AVG", "OBP", "SLG", "OPS", "BABIP", "BA/RISP", "PS/PA"]:
            if c in out.columns:
                format_dict[c] = lambda x, _f=_dot3: "" if pd.isna(x) else _f(x)
        # Percent columns are stored as fractions (0–1) in series path
        for c in pct_cols:
            if c in out.columns:
                format_dict[c] = lambda x: "" if pd.isna(x) else f"{x * 100:.2f}%"

    # Pitching special formats
    if tab_name == "Pitching":
        for c in ["ERA", "IP", "WHIP", "BB/INN"]:
            if c in out.columns:
                format_dict[c] = lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
        # R and K-L as integers
        for c in ["R", "K-L"]:
            if c in out.columns:
                format_dict[c] = lambda x: "" if pd.isna(x) else f"{int(round(x))}"
        # Percent columns are already 0–100 in series path
        for c in pct_cols:
            if c in out.columns:
                format_dict[c] = lambda x: "" if pd.isna(x) else f"{float(x):.2f}%"

    # Catching CS% as 2 decimal number (no percent sign)
    if tab_name == "Catching" and "CS%" in out.columns:
        format_dict["CS%"] = lambda x: "" if pd.isna(x) else f"{float(x):.2f}"

    # Int-like columns for this tab
    for c in int_like:
        if c in out.columns and c not in format_dict:
            format_dict[c] = lambda x: "" if pd.isna(x) else f"{int(round(x))}"

    # Default numeric formatting for all other numeric columns
    for c in out.columns:
        if c in format_dict:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            format_dict[c] = lambda x: "" if pd.isna(x) else f"{float(x):.3f}"

    styled = out.style.format(format_dict)
    if "Last" in out.columns:
        styled = styled.apply(_bold_totals(out), axis=1)

    return styled, {}
    
def _format_cumulative(df: pd.DataFrame, tab_name: str):
    """Format cumulative season stats via Styler.format without breaking numeric sort."""
    if df is None or df.empty:
        return df, {}

    out = df.copy()
    pct_cols = [c for c in out.columns if isinstance(c, str) and c.endswith("%")]
    int_like = set(INT_LIKE_BY_TAB.get(tab_name, []))
    format_dict = {}

    # Percent columns already stored as 0–100 in cumulative
    for c in pct_cols:
        if c in out.columns:
            format_dict[c] = lambda x: "" if pd.isna(x) else f"{float(x):.2f}%"

    # Hitting rate stats as .xxx
    if tab_name == "Hitting":
        for c in ["AVG", "OBP", "SLG", "OPS", "BABIP", "BA/RISP", "PS/PA"]:
            if c in out.columns:
                format_dict[c] = lambda x, _f=_dot3: "" if pd.isna(x) else _f(x)

    # Pitching special formats
    if tab_name == "Pitching":
        for c in ["ERA", "IP", "WHIP", "BB/INN"]:
            if c in out.columns:
                format_dict[c] = lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
        if "BA/RISP" in out.columns:
            format_dict["BA/RISP"] = lambda x: "" if pd.isna(x) else f"{float(x):.3f}"
        for c in ["R", "K-L"]:
            if c in out.columns:
                format_dict[c] = lambda x: "" if pd.isna(x) else f"{int(round(x))}"

    # Int-like columns for this tab
    for c in int_like:
        if c in out.columns and c not in format_dict:
            format_dict[c] = lambda x: "" if pd.isna(x) else f"{int(round(x))}"

    # Default numeric formatting for all other numeric columns
    for c in out.columns:
        if c in format_dict:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            format_dict[c] = lambda x: "" if pd.isna(x) else f"{float(x):.3f}"

    styled = out.style.format(format_dict)
    if "Last" in out.columns:
        styled = styled.apply(_bold_totals(out), axis=1)

    return styled, {}

# -------------------------------------------------------------------
# DATA SOURCE ADAPTERS AND PIPELINES
# -------------------------------------------------------------------
def list_series_csvs():
    """List non cumulative CSV base names in working dir."""
    names = []
    for p in glob.glob("*.csv"):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.lower() != os.path.splitext(CUMULATIVE_FILE)[0].lower():
            names.append(base)
    return sorted(names)

def _read_cumulative_csv() -> pd.DataFrame:
    """Locate and read cumulative.csv, clean, and return."""
    candidates = ["cumulative.csv", "/mnt/data/cumulative.csv"]
    candidates += [p for p in glob.glob("*.csv") + glob.glob("/mnt/data/*.csv") if "cumulative" in os.path.basename(p).lower()]
    for path in candidates:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, header=1, dtype=str)
                df = df.applymap(lambda x: x.strip().replace('"', '') if isinstance(x, str) else x)
                df = df.replace({"-": np.nan, "": np.nan, "N/A": np.nan})
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="ignore")
                return clean_df(df)
        except Exception as e:
            st.warning(f"Failed reading {path}: {e}")
    return pd.DataFrame()

def build_hitting_from_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    """Build hitting table from cumulative export."""
    return prepare_batting_stats(raw_all)

def build_pitching_from_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    """Build pitching table from cumulative export."""
    return prepare_pitching_stats(raw_all, from_cumulative=True)

def build_fielding_from_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    """Build fielding table from cumulative export."""
    return prepare_fielding_stats(raw_all)

def build_catching_from_cumulative(raw_all: pd.DataFrame) -> pd.DataFrame:
    """Build catching table from cumulative export."""
    return prepare_catching_stats(raw_all)

def build_hitting_from_series(selected) -> pd.DataFrame:
    """Build hitting table by aggregating selected series."""
    return prepare_batting_stats(generate_aggregated_hitting_df(aggregate_stats_hitting(selected)))

def build_pitching_from_series(selected) -> pd.DataFrame:
    """Build pitching table by aggregating selected series."""
    pitch_df = prepare_pitching_stats(generate_aggregated_pitching_df(aggregate_stats_pitching(selected)))
    if "IP" in pitch_df.columns:
        min_ip = QUAL_MINS.get("Pitching", 0.1)
        pitch_df = pitch_df[pitch_df["IP"].fillna(0) >= min_ip].reset_index(drop=True)
    return pitch_df

def build_fielding_from_series(selected) -> pd.DataFrame:
    """Build fielding table by aggregating selected series."""
    return prepare_fielding_stats(aggregate_stats_fielding(selected))

def build_catching_from_series(selected) -> pd.DataFrame:
    """Build catching table by aggregating selected series."""
    return prepare_catching_stats(clean_df(aggregate_stats_catching(selected)))

BUILDERS = {
    "Cumulative": {
        "Hitting":  build_hitting_from_cumulative,
        "Pitching": build_pitching_from_cumulative,
        "Fielding": build_fielding_from_cumulative,
        "Catching": build_catching_from_cumulative,
    },
    "Series": {
        "Hitting":  build_hitting_from_series,
        "Pitching": build_pitching_from_series,
        "Fielding": build_fielding_from_series,
        "Catching": build_catching_from_series,
    },
}

def get_frames_from_cumulative(stat_types):
    """Get dict of stat DataFrames from cumulative.csv, filtered by qualifiers."""
    raw_all = _read_cumulative_csv()
    if raw_all.empty:
        st.error("No valid cumulative CSV found.")
        return {s: pd.DataFrame() for s in stat_types}
    frames = {s: BUILDERS["Cumulative"][s](raw_all) for s in stat_types}
    return _apply_qual_mins(frames)

def get_frames_from_series(stat_types, selected_series):
    """Get dict of stat DataFrames for series mode."""
    return {s: BUILDERS["Series"][s](selected_series) for s in stat_types}

def _apply_qual_mins(frames: dict) -> dict:
    """Apply minimum thresholds for PA, IP, TC, INN by stat type."""
    out = {}
    for key, df in frames.items():
        if df is None or df.empty:
            out[key] = df
            continue
        dfx = df.copy()
        if key == "Hitting" and "PA" in dfx.columns:
            dfx = dfx[dfx["PA"] >= QUAL_MINS["Hitting"]]
        elif key == "Pitching" and "IP" in dfx.columns:
            dfx = dfx[dfx["IP"] >= QUAL_MINS["Pitching"]]
        elif key == "Fielding" and "TC" in dfx.columns:
            dfx = dfx[dfx["TC"] >= QUAL_MINS["Fielding"]]
        elif key == "Catching" and "INN" in dfx.columns:
            dfx = dfx[dfx["INN"] >= QUAL_MINS["Catching"]]
        out[key] = dfx
    return out

def extract_all_players(frames: dict):
    """Collect set of all last names across frames for player filter."""
    names = set()
    for df in frames.values():
        if df is not None and not df.empty and "Last" in df.columns:
            names.update(df["Last"].dropna().astype(str))
    return sorted(names)

def filter_players(df: pd.DataFrame, selected_lastnames):
    """Filter DataFrame to selected last names (no filter if empty selection)."""
    if not selected_lastnames or "Last" not in df.columns:
        return df
    return df[df["Last"].isin(selected_lastnames)].copy()

# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------
st.set_page_config(page_title="EUCB Stats (Fall 2025)", layout="wide")
st.title("EUCB Stats (Fall 2025)")

with st.sidebar:
    st.header("Filters")
    source_mode = st.radio(
        "Data source",
        ["Cumulative", "Series"],
        index=0,
        help="Cumulative uses cumulative.csv, Series uses selected game CSVs."
    )
    stat_types = st.multiselect(
        "Stat type(s)",
        STAT_TYPES_ALL,
        default=STAT_TYPES_ALL,
        help="Choose which player groups to display."
    )
    series_options = list_series_csvs()
    selected_series = []
    if source_mode == "Series":
        selected_series = st.multiselect(
            "Series (choose one or many)",
            options=series_options,
            default=series_options[:1] if series_options else [],
            help="Series correspond to CSV base names (for example wake, jmu, unc)."
        )

if source_mode == "Cumulative":
    frames = get_frames_from_cumulative(stat_types or STAT_TYPES_ALL)
else:
    if not selected_series:
        st.warning("Select at least one series to view stats.")
        st.stop()
    frames = get_frames_from_series(stat_types or STAT_TYPES_ALL, selected_series)

all_player_lastnames = extract_all_players(frames)
selected_players = st.multiselect(
    "Filter by player (Last name); leave empty for All",
    options=all_player_lastnames,
    default=[],
)

tabs_to_show = stat_types or STAT_TYPES_ALL
tabs = st.tabs(tabs_to_show)

for tab_name, tab in zip(tabs_to_show, tabs):
    with tab:
        df = frames.get(tab_name, pd.DataFrame())
        if df.empty:
            st.info(f"No data for **{tab_name}** with current filters.")
            continue
        df_filtered = filter_players(df, selected_players)
        df_filtered = _drop_rows_nan_names(df_filtered)
        df_filtered = _append_totals(df_filtered, tab_name)

        if selected_players and tab_name == "Pitching":
            before = len(df_filtered)
            df_filtered = _pitching_ip_gt_zero(df_filtered)
            if df_filtered.empty:
                msg = "No **Pitching** rows match selected player(s) with > 0 IP." if before > 0 else "No **Pitching** rows match selected player(s)."
                st.warning(msg)
                continue

        if tab_name == "Pitching":
            df_filtered = df_filtered.drop(columns=[c for c in ["FIP", "SB%", "BA/RISP"] if c in df_filtered.columns])

        if df_filtered.empty:
            if selected_players:
                st.warning(f"No **{tab_name}** rows match selected player(s).")
            else:
                st.info(f"No data for **{tab_name}** with current filters.")
            continue

        if source_mode == "Series":
            df_display, column_config = _format_series(df_filtered, tab_name)
        else:
            df_display, column_config = _format_cumulative(df_filtered, tab_name)

        st.subheader(f"{tab_name} Stats")
        st.dataframe(df_display, use_container_width=True, hide_index=True, column_config=column_config)

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
