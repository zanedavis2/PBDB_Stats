# eucb_stats_app.py
import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# 0) CONSTANTS & KEYS (unchanged)
# =============================================================================
STAT_TYPES_ALL = ["Hitting", "Pitching", "Fielding", "Catching"]
QUAL_MINS = {"Hitting": 1, "Pitching": 0.1, "Fielding": 1, "Catching": 0.1}
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

RATE_COLS = {
    "Hitting": ["AVG","OBP","SLG","OPS","QAB%","BB/K","C%","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","PS/PA"],
    "Pitching": ["ERA","WHIP","BB/INN","S%","FPS%","FPSO%","FPSH%","SM%","LD%","FB%","GB%","HHB%","WEAK%","BAA","BABIP","BA/RISP","SB%","<3%"],
}

# =============================================================================
# 1) DOMAIN / LOGIC LAYER (UNCHANGED CALCULATIONS)
#     -- Copied from your code with zero changes to logic
# =============================================================================
def clean_df(df):
    if "Last" in df.columns and "First" in df.columns:
        df["Last"] = df["Last"].astype(str).str.strip()
        df["First"] = df["First"].astype(str).str.strip()

        def _norm_missing(s):
            s = s.astype(str).str.strip()
            lower = s.str.lower()
            s = s.mask(lower.isin(["", "nan", "none"]))
            return s
        df["Last"] = _norm_missing(df["Last"])
        df["First"] = _norm_missing(df["First"])

        totals_idx = df.index[df["Last"].isna() & df["First"].isna()]
        if len(totals_idx) > 0:
            first_total = totals_idx[0]
            df = df.loc[: first_total - 1].reset_index(drop=True)
    return df

def prepare_batting_stats(df):
    df = df.copy()
    columns_to_keep = [
        "Last","First","PA","AB","H","AVG","OBP","SLG","OPS","RBI","R","BB","SO","XBH","2B","3B","HR",
        "TB","SB","PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB %","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI",
    ]
    existing = [c for c in columns_to_keep if c in df.columns]
    df = df[existing].copy()
    if "PA" in df.columns:
        df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
        df = df[df["PA"] != 0].reset_index(drop=True)
    if "Last" in df.columns and "First" in df.columns:
        df = df.sort_values(by=["Last","First"]).reset_index(drop=True)
    return df

def prepare_pitching_stats(df, from_cumulative=False):
    df = df.copy()
    if from_cumulative:
        df = df.iloc[:, [1, 2] + list(range(53, 148))]
        df.columns = [c.replace(".1", "") for c in df.columns]
    columns_to_keep = [
        "Last","First","IP","ERA","WHIP","H","R","ER","BB","BB/INN","SO","K-L","HR",
        "S%","FPS%","FPSO%","FPSH%","SM%","<3%","LD%","FB%","GB%","HHB%","WEAK%","BBS","BAA","BABIP",
        "BA/RISP","CS","SB","SB%","FIP"
    ]
    existing = [c for c in columns_to_keep if c in df.columns]
    df = df[existing].copy()
    if "IP" in df.columns:
        df["IP"] = pd.to_numeric(df["IP"], errors="coerce")
        df = df[df["IP"] != 0].reset_index(drop=True)
    for col in df.columns:
        if col not in ["Last","First","BABIP","BAA","BA/RISP"]:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(2)
    if "Last" in df.columns and "First" in df.columns:
        df = df.sort_values(by=["Last","First"]).reset_index(drop=True)
    return df

def prepare_fielding_stats(df):
    df = df.copy()
    columns_to_keep = ["Last","First","TC","A","PO","FPCT","E","DP"]
    existing = [c for c in columns_to_keep if c in df.columns]
    df = df[existing].copy()
    if "TC" in df.columns:
        df["TC"] = pd.to_numeric(df["TC"], errors="coerce")
        df = df[df["TC"] != 0].reset_index(drop=True)
    if "Last" in df.columns and "First" in df.columns:
        df = df.sort_values(by=["Last","First"]).reset_index(drop=True)
    for col in df.columns:
        if col not in ["Last","First","FPCT"]:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(0)
    return df

def prepare_catching_stats(df):
    df = df.copy()
    columns_to_keep = ["Last","First","INN","PB","SB-ATT","CS","CS%"]
    existing = [c for c in columns_to_keep if c in df.columns]
    df = df[existing].copy()
    if "INN" in df.columns:
        df["INN"] = pd.to_numeric(df["INN"], errors="coerce")
        df = df[df["INN"] != 0].reset_index(drop=True)
    return df

def filter_by_player(player_list, df):
    if "Last" not in df.columns:
        raise ValueError("DataFrame must contain a 'Last' column.")
    return df[df["Last"].isin(player_list)].copy()

def aggregate_stats_pitching(csv_files):
    cols_to_keep = [
        "IP","ER","H","BB","R","SO","K-L","HR","#P","BF","HBP","FPS%","FPSO%","FPSW%","FPSH%","S%","SM%",
        "LD%","FB%","GB%","BABIP","BA/RISP","CS","SB","SB%","<3%","HHB%","WEAK%","BBS",
    ]
    dfs = []
    for name in csv_files:
        file = f"{name}.csv"
        df = pd.read_csv(file, header=1)
        df = df.iloc[:, [1, 2] + list(range(53, 148))]
        df.columns = [c.replace(".1", "") for c in df.columns]

        if "Last" not in df.columns: df["Last"] = ""
        if "First" not in df.columns: df["First"] = ""

        df = df[[c for c in cols_to_keep + ["Last","First"] if c in df.columns]]
        df["Last"]  = df["Last"].astype(str).str.strip().str.title()
        df["First"] = df["First"].astype(str).str.strip().str.title()

        for col in df.columns:
            if col not in ["Last","First"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        def convert_innings(ip):
            try:
                whole = int(ip)
                fraction = round((ip - whole) * 10)
                if fraction == 1: return whole + 1/3
                elif fraction == 2: return whole + 2/3
                else: return float(ip)
            except Exception:
                return float("nan")

        if "IP" in df.columns:
            df["IP"] = df["IP"].apply(convert_innings)

        df["Strikes"]           = (df["S%"]    * df["#P"] / 100).round(0).astype(int)
        df["FirstPitchStrikes"] = (df["FPS%"]  * df["BF"] / 100).round(0).astype(int)
        df["FPSO"]              = (df["FPSO%"] * df["BF"] / 100).round(0).astype(int)
        df["FPSH"]              = (df["FPSH%"] * df["BF"] / 100).round(0).astype(int)

        total_bip = df["BF"] - df["SO"] - df["BB"] - df["HBP"]
        df["GroundBalls"]  = (df["GB%"]  * total_bip / 100).round(0).astype(int)
        df["FlyBalls"]     = (df["FB%"]  * total_bip / 100).round(0).astype(int)
        df["LineDrives"]   = (df["LD%"]  * total_bip / 100).round(0).astype(int)
        df["HardHitBalls"] = (df["HHB%"] * total_bip / 100).round(0).astype(int)
        df["WeakContact"]  = (df["WEAK%"]* total_bip / 100).round(0).astype(int)
        df["Under3Pitches"]= (df["<3%"]  * df["BF"] / 100).round(0).astype(int)
        df["SwingMisses"]  = (df["SM%"]  * df["#P"] / 100).round(0).astype(int)

        drop_cols = [c for c in df.columns if c.endswith("%")]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for col in agg_df.columns:
        if col not in ["Last","First"]:
            agg_df[col] = agg_df[col].fillna(0).round(3)
    return agg_df

def generate_aggregated_pitching_df(df):
    needed_counts = [
        "IP","ER","H","BB","R","SO","K-L","HR","#P","BF","HBP","Strikes","FirstPitchStrikes","FPSO","FPSH",
        "GroundBalls","FlyBalls","LineDrives","HardHitBalls","WeakContact","Under3Pitches","SwingMisses","BBS","CS","SB"
    ]
    df = df.copy()
    for c in needed_counts:
        if c not in df.columns: df[c] = 0

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

    if "BA/RISP" not in df.columns: df["BA/RISP"] = 0.000

    internal_cols = {
        "_IP":"IP","_ER":"ER","_H":"H","_BB":"BB","_HR":"HR","_SO":"SO",
        "_NP":"#P","_BF":"BF","_HBP":"HBP","_STR":"Strikes","_FPS":"FirstPitchStrikes",
        "_FPSO":"FPSO","_FPSH":"FPSH","_GB":"GroundBalls","_FB":"FlyBalls","_LD":"LineDrives",
        "_HHB":"HardHitBalls","_WEAK":"WeakContact","_U3":"Under3Pitches","_SM":"SwingMisses",
        "_BBS":"BBS","_CS":"CS","_SB":"SB"
    }
    for new, old in internal_cols.items():
        df[new] = df[old].fillna(0)

    columns_to_keep = [
        "Last","First","IP","ERA","WHIP","SO","K-L","H","R","ER","BB","BB/INN","FIP","S%","FPS%","FPSO%","FPSH%",
        "BAA","BBS","SM%","LD%","FB%","GB%","BABIP","BA/RISP","CS","SB","SB%","<3%","HHB%","WEAK%"
    ] + list(internal_cols.keys())
    for c in columns_to_keep:
        if c not in df.columns: df[c] = 0
    return df[columns_to_keep].copy()

def aggregate_stats_hitting(csv_files):
    cols_to_keep = [
        "Last","First","PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","2B","3B","HR","SB","CS",
        "QAB","HHB","LD%","FB%","GB%","H_RISP","AB_RISP","PS","2OUTRBI","XBH",
    ]
    def _pct_to_ratio(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return np.where(s > 1.0, s / 100.0, s)

    dfs = []
    for name in csv_files:
        file = f"{name}.csv"
        df = pd.read_csv(file, header=1)
        df = df[[c for c in cols_to_keep if c in df.columns]].copy()

        df["Last"]  = df["Last"].astype(str).str.strip().str.title()
        df["First"] = df["First"].astype(str).str.strip().str.title()

        for col in df.columns:
            if col not in ["Last","First"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        ld_ratio = _pct_to_ratio(df.get("LD%", 0))
        gb_ratio = _pct_to_ratio(df.get("GB%", 0))
        fb_ratio = _pct_to_ratio(df.get("FB%", 0))

        df["LD"] = np.rint(ld_ratio * df.get("AB", 0)).astype(int)
        df["GB"] = np.rint(gb_ratio * df.get("AB", 0)).astype(int)
        df["FB"] = np.rint(fb_ratio * df.get("AB", 0)).astype(int)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for c in ["LD","GB","FB"]:
        if c in agg_df.columns:
            agg_df[c] = agg_df[c].astype(int)
    return agg_df

def generate_aggregated_hitting_df(df):
    cols = [
        "Last","First","PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","2B","3B","HR","SB","CS",
        "QAB","HHB","LD","FB","GB","H_RISP","AB_RISP","PS","2OUTRBI","XBH",
    ]
    for c in cols:
        if c not in df.columns: df[c] = 0
    df = df[cols].copy()
    for c in df.columns:
        if c not in ["Last","First"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    agg_df = df.groupby(["Last","First"], as_index=False).sum()

    agg_df["AVG"]   = np.where(agg_df["AB"] > 0, agg_df["H"]/agg_df["AB"], 0)
    agg_df["OBP"]   = np.where((agg_df["AB"]+agg_df["BB"]+agg_df["HBP"]+agg_df["SF"]) > 0,
                               (agg_df["H"]+agg_df["BB"]+agg_df["HBP"])/(agg_df["AB"]+agg_df["BB"]+agg_df["HBP"]+agg_df["SF"]), 0)
    agg_df["SLG"]   = np.where(agg_df["AB"] > 0, agg_df["TB"]/agg_df["AB"], 0)
    agg_df["OPS"]   = agg_df["OBP"] + agg_df["SLG"]
    agg_df["QAB%"]  = np.where(agg_df["PA"] > 0, agg_df["QAB"]/agg_df["PA"], 0)
    agg_df["BB/K"]  = np.where(agg_df["SO"] > 0, agg_df["BB"]/agg_df["SO"], agg_df["BB"])
    agg_df["C%"]    = np.where(agg_df["AB"] > 0, 1 - (agg_df["SO"]/agg_df["AB"]), 0)
    agg_df["HHB%"]  = np.where(agg_df["AB"] > 0, agg_df["HHB"]/agg_df["AB"], 0)

    total_batted = agg_df["LD"] + agg_df["FB"] + agg_df["GB"]
    agg_df["LD%"]  = np.where(total_batted > 0, agg_df["LD"]/total_batted, 0)
    agg_df["FB%"]  = np.where(total_batted > 0, agg_df["FB"]/total_batted, 0)
    agg_df["GB%"]  = np.where(total_batted > 0, agg_df["GB"]/total_batted, 0)

    denom = agg_df["AB"] - agg_df["SO"] - agg_df["HR"] + agg_df["SF"]
    agg_df["BABIP"]  = np.where(denom > 0, (agg_df["H"] - agg_df["HR"]) / denom, 0)
    agg_df["BA/RISP"] = np.where(agg_df["AB_RISP"] > 0, agg_df["H_RISP"] / agg_df["AB_RISP"], 0)
    agg_df["PS/PA"]  = np.where(agg_df["PA"] > 0, agg_df["PS"] / agg_df["PA"], 0)

    pct_cols = ["AVG","OBP","SLG","OPS","QAB%","BB/K","C%","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","PS/PA"]
    agg_df[pct_cols] = agg_df[pct_cols].round(3)

    final_cols = [
        "Last","First","PA","AB","AVG","OBP","OPS","SLG","H","R","RBI","BB","2B","3B","HR","SB",
        "QAB","QAB%","BB/K","C%","HHB","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI","XBH","TB","PS/PA","SO",
    ]
    return agg_df[[c for c in final_cols if c in agg_df.columns]]

def aggregate_stats_fielding(csv_files):
    cols_to_keep = ["TC","A","PO","E","DP"]
    dfs = []
    for name in csv_files:
        file = f"{name}.csv"
        try:
            df = pd.read_csv(file, header=1)
            df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
            df.columns = [c.replace(".1", "") for c in df.columns]

            if "Last" not in df.columns: df["Last"] = ""
            if "First" not in df.columns: df["First"] = ""

            df = df[[c for c in cols_to_keep + ["Last","First"] if c in df.columns]]
            df["Last"]  = df["Last"].astype(str).str.strip().str.title()
            df["First"] = df["First"].astype(str).str.strip().str.title()
            for col in df.columns:
                if col not in ["Last","First"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=cols_to_keep + ["Last","First"])
    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for col in agg_df.columns:
        if col not in ["Last","First"]:
            agg_df[col] = agg_df[col].fillna(0).round(3)
    agg_df["FPCT"] = ((agg_df["A"] + agg_df["PO"]) / agg_df["TC"]).round(3).fillna(0)
    return agg_df

def aggregate_stats_catching(csv_files):
    cols_to_keep = ["INN","PB","SB","SB-ATT","CS"]
    dfs = []
    for name in csv_files:
        file = f"{name}.csv"
        try:
            df = pd.read_csv(file, header=1)
            df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
            df.columns = [c.replace(".1", "") for c in df.columns]
            df.columns = [c.replace(".2", "") for c in df.columns]

            if "Last" not in df.columns: df["Last"] = ""
            if "First" not in df.columns: df["First"] = ""

            df = df[[c for c in cols_to_keep + ["Last","First"] if c in df.columns]]
            df["Last"]  = df["Last"].astype(str).str.strip().str.title()
            df["First"] = df["First"].astype(str).str.strip().str.title()

            for col in df.columns:
                if col not in ["Last","First","SB-ATT"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame(columns=cols_to_keep + ["Last","First"])

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

    agg_df = combined.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    for col in agg_df.columns:
        if col not in ["Last","First","INN"]:
            agg_df[col] = agg_df[col].fillna(0).round(0)

    agg_df["CS%"]   = np.where(agg_df["ATT"] > 0, (agg_df["CS"] / agg_df["ATT"] * 100).round(1), 0)
    agg_df["SB-ATT"] = agg_df["SB"].astype(int).astype(str) + "-" + agg_df["ATT"].astype(int).astype(str)
    return agg_df.drop(columns=["SB","ATT"])

def _drop_rows_nan_names(df):
    if df is None or df.empty: return df
    for c in [col for col in ["Last","First"] if col in df.columns]:
        s = df[c].astype(str).str.strip()
        df[c] = s.mask(s.str.lower().isin(["","nan","none"]))
    cols = [c for c in ["Last","First"] if c in df.columns]
    if not cols: return df
    return df.dropna(subset=cols, how="all").reset_index(drop=True)

def _append_totals(df, tab_name):
    if df is None or df.empty: return df
    base = df.copy()
    if "Last" in base.columns:
        lower_last = base["Last"].astype(str).str.strip().str.lower()
        if lower_last.isin(["totals","total",""]).any():
            base["_is_total"] = lower_last.isin(["totals","total",""])
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

        for raw_col in ["PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","HR","QAB","PS","SB","XBH","2B","3B","H_RISP","AB_RISP"]:
            if raw_col in base.columns: totals[raw_col] = ssum(raw_col)

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
        for raw in ["IP","ER","H","BB","HR","SO","BF","HBP","SB","CS","#P"]:
            if raw in base.columns: totals[raw] = ssum(raw)
        IP, ER, Hh, BBh, HRh, SOh = totals.get("IP",0), totals.get("ER",0), totals.get("H",0), totals.get("BB",0), totals.get("HR",0), totals.get("SO",0)
        BF, HBP, SB, CS, NP = totals.get("BF",0), totals.get("HBP",0), totals.get("SB",0), totals.get("CS",0), totals.get("#P",0)

        totals["ERA"]    = round((ER * 9 / IP), 2) if IP else 0
        totals["WHIP"]   = round((BBh + Hh) / IP, 2) if IP else 0
        totals["BB/INN"] = round(BBh / IP, 2) if IP else 0
        totals["FIP"]    = round(((13 * HRh + 3 * BBh - 2 * SOh) / IP) + 3.1, 2) if IP else 0
        totals["SB%"]    = round((SB / (SB + CS) * 100), 2) if (SB + CS) else 0
        totals["BAA"]    = round(Hh / (BF - BBh - HBP), 3) if (BF - BBh - HBP) > 0 else 0
        totals["BABIP"]  = round((Hh - HRh) / (BF - SOh - HRh - BBh - HBP), 3) if (BF - SOh - HRh - BBh - HBP) > 0 else 0

        for c in base.columns:
            if c in ["Last","First"] or c in totals: continue
            if isinstance(c, str) and c.endswith("%"): totals[c] = round(smean(c), 2)
            elif pd.api.types.is_numeric_dtype(base[c]): totals[c] = ssum(c)

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
            totals["CS%"] = round((totals.get("CS",0) / att_sum * 100), 1) if att_sum else 0

    for c in base.columns:
        if c in totals or c in ["Last","First"]: continue
        if isinstance(c, str) and c.endswith("%"): totals[c] = round(smean(c), 3)
        elif pd.api.types.is_numeric_dtype(base[c]): totals[c] = ssum(c)
        else: totals[c] = ""

    totals_df = pd.DataFrame([totals]).reindex(columns=base.columns)
    if "Last" in base.columns:
        mask = base["Last"].astype(str).str.strip().str.lower().isin(["totals","total"])
        base = base[~mask]
    return pd.concat([base, totals_df], ignore_index=True)

def _pitching_ip_gt_zero(df):
    if "IP" not in df.columns: return df
    return df[df["IP"].fillna(0) > 0].copy()


def _format_series(df, tab_name):
    """
    SERIES VIEW formatter (aggregated series → ratios for % need *100).
    - Scale % columns by 100, render with 2 decimals + '%'
    - Hitting rates as .xxx with 3 decimals
    - Pitching ERA always 2 decimals; R and K-L as ints
    - Other numerics: 3 fixed decimals (no stripping)
    """
    if df is None or df.empty:
        return df, {}

    out = df.copy()

    # Helpers
    def _dot3(x):
        if pd.isna(x):
            return ""
        s = f"{float(x):.3f}"
        if s.startswith("0."):
            return "." + s[2:]
        if s.startswith("-0."):
            return "-." + s[3:]
        return s

    pct_cols = [c for c in out.columns if isinstance(c, str) and c.endswith("%")]

    # SERIES: scale ratios → percent
    for c in pct_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce") * 100.0
        out[c] = out[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

    # Hitting decimals as .xxx with 3 places
    if tab_name == "Hitting":
        for c in [k for k in ["AVG","OBP","SLG","OPS","BABIP","BA/RISP","PS/PA"] if k in out.columns]:
            out[c] = out[c].map(_dot3)

    # Pitching specifics
    if tab_name == "Pitching":
        if "ERA" in out.columns:
            out["ERA"] = out["ERA"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")
        for c in [k for k in ["R","K-L"] if k in out.columns]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("Int64").astype(str).replace("<NA>", "")

    # Int-like columns per tab (display as ints)
    int_like_by_tab = {
        "Hitting":  ["PA","AB","H","R","RBI","BB","SO","2B","3B","HR","SB","QAB","XBH","TB","2OUTRBI","H_RISP","AB_RISP","HHB"],
        "Pitching": ["H","R","ER","BB","SO","HR","BBS","CS","SB","K-L","BF","#P","HBP",
                     "GroundBalls","FlyBalls","LineDrives","HardHitBalls","WeakContact","Under3Pitches","SwingMisses"],
        "Fielding": ["TC","A","PO","E","DP"],
        "Catching": ["INN","PB","CS"],  # SB-ATT stays text
    }
    int_like = set(int_like_by_tab.get(tab_name, []))

    for c in out.columns:
        if c in pct_cols:
            continue
        if c in int_like:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("Int64").astype(str).replace("<NA>", "")
        else:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")

    # Bold totals row
    if "Last" in out.columns:
        def _bold(row):
            return ["font-weight: bold" if str(row.get("Last","")).strip().lower() in {"totals","total"} else "" for _ in row]
        out = out.style.apply(_bold, axis=1)

    return out, {}



def _format_cumulative(df, tab_name):
    """
    CUMULATIVE VIEW formatter (cumulative.csv → % already in percent units).
    - Do NOT scale % columns; just render with 2 decimals + '%'
    - Hitting rates as .xxx with 3 decimals
    - Pitching ERA always 2 decimals; R and K-L as ints
    - Other numerics: 3 fixed decimals (no stripping)
    """
    if df is None or df.empty:
        return df, {}

    out = df.copy()

    def _dot3(x):
        if pd.isna(x):
            return ""
        s = f"{float(x):.3f}"
        if s.startswith("0."):
            return "." + s[2:]
        if s.startswith("-0."):
            return "-." + s[3:]
        return s

    pct_cols = [c for c in out.columns if isinstance(c, str) and c.endswith("%")]

    # CUMULATIVE: already percent → format only
    for c in pct_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

    if tab_name == "Hitting":
        for c in [k for k in ["AVG","OBP","SLG","OPS","BABIP","BA/RISP","PS/PA"] if k in out.columns]:
            out[c] = out[c].map(_dot3)

    if tab_name == "Pitching":
        if "ERA" in out.columns:
            out["ERA"] = out["ERA"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")
        for c in [k for k in ["R","K-L"] if k in out.columns]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("Int64").astype(str).replace("<NA>", "")

    int_like_by_tab = {
        "Hitting":  ["PA","AB","H","R","RBI","BB","SO","2B","3B","HR","SB","QAB","XBH","TB","2OUTRBI","H_RISP","AB_RISP","HHB"],
        "Pitching": ["H","R","ER","BB","SO","HR","BBS","CS","SB","K-L","BF","#P","HBP",
                     "GroundBalls","FlyBalls","LineDrives","HardHitBalls","WeakContact","Under3Pitches","SwingMisses"],
        "Fielding": ["TC","A","PO","E","DP"],
        "Catching": ["INN","PB","CS"],
    }
    int_like = set(int_like_by_tab.get(tab_name, []))

    for c in out.columns:
        if c in pct_cols:
            continue
        if c in int_like:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("Int64").astype(str).replace("<NA>", "")
        else:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")

    if "Last" in out.columns:
        def _bold(row):
            return ["font-weight: bold" if str(row.get("Last","")).strip().lower() in {"totals","total"} else "" for _ in row]
        out = out.style.apply(_bold, axis=1)

    return out, {}






# =============================================================================
# 2) DATA-SOURCE ADAPTERS
#     One place to change how data gets loaded/normalized per source
# =============================================================================
def list_series_csvs():
    names = []
    for p in glob.glob("*.csv"):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.lower() != os.path.splitext(CUMULATIVE_FILE)[0].lower():
            names.append(base)
    return sorted(names)

def _read_cumulative_csv():
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

# =============================================================================
# 3) STAT-TYPE PIPELINES (single place each stat is built for BOTH sources)
#     - Each returns a ready-to-display DataFrame for that stat type.
# =============================================================================
def build_hitting_from_cumulative(raw_all):
    return prepare_batting_stats(raw_all)

def build_pitching_from_cumulative(raw_all):
    return prepare_pitching_stats(raw_all, from_cumulative=True)

def build_fielding_from_cumulative(raw_all):
    return prepare_fielding_stats(raw_all)

def build_catching_from_cumulative(raw_all):
    return prepare_catching_stats(raw_all)

def build_hitting_from_series(selected):
    _hit = aggregate_stats_hitting(selected)
    return prepare_batting_stats(generate_aggregated_hitting_df(_hit))

def build_pitching_from_series(selected):
    _pit = aggregate_stats_pitching(selected)
    pitch_df = prepare_pitching_stats(generate_aggregated_pitching_df(_pit))
    if "IP" in pitch_df.columns:
        min_ip = QUAL_MINS.get("Pitching", 0.1)
        pitch_df = pitch_df[pitch_df["IP"].fillna(0) >= min_ip].reset_index(drop=True)
    return pitch_df

def build_fielding_from_series(selected):
    _fld = aggregate_stats_fielding(selected)
    return prepare_fielding_stats(_fld)

def build_catching_from_series(selected):
    _cat = aggregate_stats_catching(selected)
    return prepare_catching_stats(clean_df(_cat))

# A single registry that defines how to build each table per source.
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

# =============================================================================
# 4) UNIFIED PIPELINE APIS (source-agnostic)
# =============================================================================
def get_frames_from_cumulative(stat_types):
    raw_all = _read_cumulative_csv()
    if raw_all.empty:
        st.error("No valid cumulative CSV found.")
        return {s: pd.DataFrame() for s in stat_types}
    frames = {s: BUILDERS["Cumulative"][s](raw_all) for s in stat_types}
    return _apply_qual_mins(frames)

def get_frames_from_series(stat_types, selected_series):
    frames = {s: BUILDERS["Series"][s](selected_series) for s in stat_types}
    return frames  # series path already filters pitching IP; others same as your code

def _apply_qual_mins(frames):
    out = {}
    for key, df in frames.items():
        if df is None or df.empty:
            out[key] = df
            continue
        dfx = df.copy()
        if key == "Hitting" and "PA" in dfx.columns:
            dfx = dfx[dfx["PA"] >= QUAL_MINS.get("Hitting", 0)]
        elif key == "Pitching" and "IP" in dfx.columns:
            dfx = dfx[dfx["IP"] >= QUAL_MINS.get("Pitching", 0)]
        elif key == "Fielding" and "TC" in dfx.columns:
            dfx = dfx[dfx["TC"] >= QUAL_MINS.get("Fielding", 0)]
        elif key == "Catching" and "INN" in dfx.columns:
            dfx = dfx[dfx["INN"] >= QUAL_MINS.get("Catching", 0)]
        out[key] = dfx
    return out

def extract_all_players(frames):
    names = set()
    for df in frames.values():
        if df is not None and not df.empty and "Last" in df.columns:
            names.update(df["Last"].dropna().astype(str))
    return sorted(names)

def filter_players(df, selected_lastnames):
    if not selected_lastnames: return df
    if "Last" not in df.columns: return df
    return df[df["Last"].isin(selected_lastnames)].copy()

# =============================================================================
# 5) UI
# =============================================================================
st.set_page_config(page_title="EUCB Stats (Fall 2025)", layout="wide")
st.title("EUCB Stats (Fall 2025)")

with st.sidebar:
    st.header("Filters")

    source_mode = st.radio(
        "Data source",
        ["Cumulative", "Series"],
        index=0,
        help="Cumulative shows season-to-date from cumulative.csv. Series lets you pick one or more series CSVs."
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
            help="Series correspond to CSV base names (e.g., wake, jmu, unc)."
        )

# Load frames in a single call through the unified pipeline
if source_mode == "Cumulative":
    frames = get_frames_from_cumulative(stat_types if stat_types else STAT_TYPES_ALL)
else:
    if not selected_series:
        st.warning("Select at least one series to view stats.")
        st.stop()
    frames = get_frames_from_series(stat_types if stat_types else STAT_TYPES_ALL, selected_series)

all_player_lastnames = extract_all_players(frames)
selected_players = st.multiselect(
    "Filter by player (Last name); leave empty for All",
    options=all_player_lastnames,
    default=[],
)

tabs_to_show = stat_types if stat_types else STAT_TYPES_ALL
tabs = st.tabs(tabs_to_show)

for tab_name, tab in zip(tabs_to_show, tabs):
    with tab:
        df = frames.get(tab_name, pd.DataFrame())
        if df.empty:
            st.info(f"No data for **{tab_name}** with current filters.")
            continue

        df_filtered = filter_players(df, selected_players)
        df_filtered = _drop_rows_nan_names(df_filtered)

        # Append totals row prior to formatting (same logic)
        df_filtered = _append_totals(df_filtered, tab_name)

        # Optional IP>0 guard for Pitching after user filtering (same as your code)
        if selected_players and tab_name == "Pitching":
            df_before = len(df_filtered)
            df_filtered = _pitching_ip_gt_zero(df_filtered)
            if df_filtered.empty:
                st.warning(
                    "No **Pitching** rows match selected player(s) with > 0 IP."
                    if df_before > 0 else
                    "No **Pitching** rows match selected player(s)."
                )
                continue

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
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
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
