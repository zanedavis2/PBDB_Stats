import os
import io
import glob
import numpy as np
import pandas as pd
import streamlit as st

# --- Guards for constants used by the UI (avoid NameError if moved around) ---
try:
    STAT_TYPES_ALL
except NameError:
    STAT_TYPES_ALL = ["Hitting", "Pitching", "Fielding", "Catching"]

try:
    QUAL_MINS
except NameError:
    QUAL_MINS = {"Hitting": 0, "Pitching": 0.0, "Fielding": 0, "Catching": 0.0}

"""
This Streamlit app wraps the original PBDB stats functions in a single-file interface
that lets you select series (e.g., ["wake", "jmu"]) or upload cumulative CSVs, and
then displays filterable tables for Hitting, Pitching, Fielding, and Catching.

The absolute majority of the logic is copied directly from your original file, with
only light edits for small bugs and to fit Streamlit I/O.
"""

# =====================
# Original helper funcs
# =====================

def clean_df(df):
    if "Last" in df.columns and "First" in df.columns:
        # Normalize blanks to NaN
        df["Last"] = df["Last"].astype(str).str.strip()
        df["First"] = df["First"].astype(str).str.strip()
        df["Last"].replace(["", "nan", "NaN", "None"], np.nan, inplace=True)
        df["First"].replace(["", "nan", "NaN", "None"], np.nan, inplace=True)

        # Drop totals/empty tail if present
        totals_idx = df.index[df["Last"].isna() & df["First"].isna()]
        if len(totals_idx) > 0:
            first_total = totals_idx[0]
            df = df.loc[: first_total - 1].reset_index(drop=True)
    return df


def prepare_batting_stats(df):
    df = df.copy()
    columns_to_keep = [
        "Last",
        "First",
        "PA",
        "AB",
        "H",
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "RBI",
        "R",
        "BB",
        "SO",
        "XBH",
        "2B",
        "3B",
        "HR",
        "TB",
        "SB",
        "PS/PA",
        "BB/K",
        "C%",
        "QAB",
        "QAB%",
        "HHB",
        "HHB %",
        "LD%",
        "FB%",
        "GB%",
        "BABIP",
        "BA/RISP",
        "2OUTRBI",
    ]
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()

    if "PA" in df.columns:
        df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
        df = df[df["PA"] != 0].reset_index(drop=True)

    if "Last" in df.columns and "First" in df.columns:
        df = df.sort_values(by=["Last", "First"]).reset_index(drop=True)
    return df


def prepare_pitching_stats(df):
    df = df.copy()
    columns_to_keep = [
        "Last",
        "First",
        "IP",
        "ERA",
        "WHIP",
        "H",
        "R",
        "ER",
        "BB",
        "BB/INN",
        "SO",
        "K-L",
        "HR",
        "S%",
        "FPS%",
        "FPSO%",
        "FPSH%",
        "SM%",
        "<3%",
        "LD%",
        "FB%",
        "GB%",
        "HHB%",
        "WEAK%",
        "BBS",
        "BAA",
        "BABIP",
        "BA/RISP",
        "CS",
        "SB",
        "SB%",
        "FIP",
    ]
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()

    if "IP" in df.columns:
        df["IP"] = pd.to_numeric(df["IP"], errors="coerce")
        df = df[df["IP"] != 0].reset_index(drop=True)

    for col in df.columns:
        if col not in ["Last", "First", "BABIP", "BAA", "BA/RISP"]:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(2)

    if "Last" in df.columns and "First" in df.columns:
        df = df.sort_values(by=["Last", "First"]).reset_index(drop=True)
    return df


def prepare_fielding_stats(df):
    df = df.copy()
    columns_to_keep = ["Last", "First", "TC", "A", "PO", "FPCT", "E", "DP"]
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()

    if "TC" in df.columns:
        df["TC"] = pd.to_numeric(df["TC"], errors="coerce")
        df = df[df["TC"] != 0].reset_index(drop=True)

    if "Last" in df.columns and "First" in df.columns:
        df = df.sort_values(by=["Last", "First"]).reset_index(drop=True)

    for col in df.columns:
        if col not in ["Last", "First", "FPCT"]:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(0)
    return df


def prepare_catching_stats(df):
    df = df.copy()
    columns_to_keep = ["Last", "First", "INN", "PB", "SB-ATT", "CS", "CS%"]
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()

    if "INN" in df.columns:
        df["INN"] = pd.to_numeric(df["INN"], errors="coerce")
        df = df[df["INN"] != 0].reset_index(drop=True)
    return df


def filter_by_player(player_list, df):
    if "Last" not in df.columns:
        raise ValueError("DataFrame must contain a 'Last' column.")
    return df[df["Last"].isin(player_list)].copy()


# ============================
# Aggregation (Pitching)
# ============================

def aggregate_stats_pitching(csv_files):
    cols_to_keep = [
        "IP",
        "ER",
        "H",
        "BB",
        "R",
        "SO",
        "K-L",
        "HR",
        "#P",
        "BF",
        "HBP",
        "FPS%",
        "FPSO%",
        "FPSW%",
        "FPSH%",
        "S%",
        "SM%",
        "LD%",
        "FB%",
        "GB%",
        "BABIP",
        "BA/RISP",
        "CS",
        "SB",
        "SB%",
        "<3%",
        "HHB%",
        "WEAK%",
        "BBS",
    ]

    dfs = []
    for name in csv_files:
        file = f"{name}.csv"
        df = pd.read_csv(file, header=1)
        df = df.iloc[:, [1, 2] + list(range(53, 148))]
        df.columns = [c.replace(".1", "") for c in df.columns]

        if "Last" not in df.columns:
            df["Last"] = ""
        if "First" not in df.columns:
            df["First"] = ""

        df = df[[c for c in cols_to_keep + ["Last", "First"] if c in df.columns]]

        df["Last"] = df["Last"].astype(str).str.strip().str.title()
        df["First"] = df["First"].astype(str).str.strip().str.title()

        for col in df.columns:
            if col not in ["Last", "First"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Convert innings format (e.g., 4.2 → 4 + 2/3)
        def convert_innings(ip):
            try:
                whole = int(ip)
                fraction = round((ip - whole) * 10)
                if fraction == 1:
                    return whole + 1 / 3
                elif fraction == 2:
                    return whole + 2 / 3
                else:
                    return float(ip)
            except Exception:
                return float("nan")

        if "IP" in df.columns:
            df["IP"] = df["IP"].apply(convert_innings)

        # Percent → counts
        df["Strikes"] = (df["S%"] * df["#P"] / 100).round(0).astype(int)
        df["FirstPitchStrikes"] = (df["FPS%"] * df["BF"] / 100).round(0).astype(int)
        df["FPSO"] = (df["FPSO%"] * df["BF"] / 100).round(0).astype(int)
        df["FPSH"] = (df["FPSH%"] * df["BF"] / 100).round(0).astype(int)

        total_batted_balls = df["BF"] - df["SO"] - df["BB"] - df["HBP"]
        df["GroundBalls"] = (df["GB%"] * total_batted_balls / 100).round(0).astype(int)
        df["FlyBalls"] = (df["FB%"] * total_batted_balls / 100).round(0).astype(int)
        df["LineDrives"] = (df["LD%"] * total_batted_balls / 100).round(0).astype(int)
        df["HardHitBalls"] = (df["HHB%"] * total_batted_balls / 100).round(0).astype(int)
        df["WeakContact"] = (df["WEAK%"] * total_batted_balls / 100).round(0).astype(int)
        df["Under3Pitches"] = (df["<3%"] * df["BF"] / 100).round(0).astype(int)
        df["SwingMisses"] = (df["SM%"] * df["#P"] / 100).round(0).astype(int)

        drop_cols = [c for c in df.columns if c.endswith("%")]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    agg_df = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)
    for col in agg_df.columns:
        if col not in ["Last", "First"]:
            agg_df[col] = agg_df[col].fillna(0).round(3)
    return agg_df


def generate_aggregated_pitching_df(df):
    """
    Takes an aggregated integer pitching dataframe and computes derived metrics.
    Returns a clean dataframe ready for display or export.
    Safe to call on data read from series or cumulative CSVs (strings, dashes, etc.).
    """
    columns_to_keep = [
        "Last", "First", "IP", "ERA", "WHIP", "K", "K-L", "H", "R", "ER", "BB", "BB/INN",
        "FIP", "S%", "FPS%", "FPSO%", "FPSH%", "BAA", "BBS", "SM%", "LD%", "FB%", "GB%",
        "BABIP", "BA/RISP", "CS", "SB", "SB%", "<3%", "HHB%", "WEAK%"
    ]

    df = df.copy()

    # ---- Safety: ensure numeric types ----
    def _to_num(s):
        return pd.to_numeric(pd.Series(s).astype(str).str.replace("-", "", regex=False), errors="coerce")

    # Convert baseball-style innings (4.1 -> 4.3333, 4.2 -> 4.6667)
    def _convert_innings_value(ip):
        try:
            if pd.isna(ip):
                return np.nan
            ip = float(str(ip))
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

    needed_numeric = [
        "IP","ER","H","BB","SO","K-L","HR","#P","BF","Strikes","FirstPitchStrikes",
        "FPSO","FPSH","GroundBalls","FlyBalls","LineDrives","HardHitBalls","WeakContact",
        "Under3Pitches","SwingMisses","BBS","CS","SB","HBP"
    ]
    for c in needed_numeric:
        if c in df.columns:
            df[c] = _to_num(df[c])
        else:
            df[c] = 0

    if "IP" in df.columns:
        df["IP"] = df["IP"].apply(_convert_innings_value)

    # Avoid /0
    df["IP"] = df["IP"].replace(0, np.nan)
    df["BF"] = df["BF"].replace(0, np.nan)
    df["#P"] = df["#P"].replace(0, np.nan)

    # Derived metrics
    df["ERA"] = (df["ER"] * 9 / df["IP"]).round(2)
    df["WHIP"] = ((df["BB"] + df["H"]) / df["IP"]).round(2)
    df["BB/INN"] = (df["BB"] / df["IP"]).round(2)
    df["FIP"] = (((13 * df["HR"]) + (3 * df["BB"]) - (2 * df["SO"])) / df["IP"] + 3.1).round(2)

    # Percentages re-formed from counts
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

    # SB%
    df["SB%"] = np.where((df["SB"] + df["CS"]) > 0, (df["SB"] / (df["SB"] + df["CS"]) * 100).round(1), 0)

    # BAA, BABIP
    df["BAA"] = np.where((df["BF"] - df["BB"] - df["HBP"]) > 0, (df["H"] / (df["BF"] - df["BB"] - df["HBP"])) .round(3), 0)
    df["BABIP"] = np.where((df["BF"] - df["SO"] - df["HR"] - df["BB"] - df["HBP"]) > 0, ((df["H"] - df["HR"]) / (df["BF"] - df["SO"] - df["HR"] - df["BB"] - df["HBP"])) .round(3), 0)

    for col in columns_to_keep:
        if col not in df.columns:
            df[col] = 0

    df = df[columns_to_keep].copy().round(3).fillna(0)
    return df


# ============================
# Aggregation (Hitting)
# ============================

def aggregate_stats_hitting(csv_files):
    cols_to_keep = [
        "Last",
        "First",
        "PA",
        "AB",
        "H",
        "BB",
        "HBP",
        "SF",
        "TB",
        "R",
        "RBI",
        "SO",
        "2B",
        "3B",
        "HR",
        "SB",
        "CS",
        "QAB",
        "HHB",
        "LD",
        "FB",
        "GB",
        "H_RISP",
        "AB_RISP",
        "PS",
        "2OUTRBI",
        "XBH",
    ]

    dfs = []
    for name in csv_files:
        file = f"{name}.csv"
        df = pd.read_csv(file, header=1)
        df = df[[c for c in cols_to_keep if c in df.columns]]

        df["Last"] = df["Last"].astype(str).str.strip().str.title()
        df["First"] = df["First"].astype(str).str.strip().str.title()

        for col in df.columns:
            if col not in ["Last", "First"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last", "First"], as_index=False).sum()
    return agg_df


def generate_aggregated_hitting_df(df):
    cols_to_keep = [
        "Last",
        "First",
        "PA",
        "AB",
        "H",
        "BB",
        "HBP",
        "SF",
        "TB",
        "R",
        "RBI",
        "SO",
        "2B",
        "3B",
        "HR",
        "SB",
        "CS",
        "QAB",
        "HHB",
        "LD",
        "FB",
        "GB",
        "H_RISP",
        "AB_RISP",
        "PS",
        "2OUTRBI",
        "XBH",
    ]

    for col in cols_to_keep:
        if col not in df.columns:
            df[col] = 0
    df = df[cols_to_keep].copy()

    for col in df.columns:
        if col not in ["Last", "First"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    agg_df = df.groupby(["Last", "First"], as_index=False).sum()

    # Derived
    agg_df["AVG"] = np.where(agg_df["AB"] > 0, agg_df["H"] / agg_df["AB"], 0)
    agg_df["OBP"] = np.where(
        (agg_df["AB"] + agg_df["BB"] + agg_df["HBP"] + agg_df["SF"]) > 0,
        (agg_df["H"] + agg_df["BB"] + agg_df["HBP"]) / (agg_df["AB"] + agg_df["BB"] + agg_df["HBP"] + agg_df["SF"]),
        0,
    )
    agg_df["SLG"] = np.where(agg_df["AB"] > 0, agg_df["TB"] / agg_df["AB"], 0)
    agg_df["OPS"] = agg_df["OBP"] + agg_df["SLG"]
    agg_df["QAB%"] = np.where(agg_df["PA"] > 0, agg_df["QAB"] / agg_df["PA"], 0)
    agg_df["BB/K"] = np.where(agg_df["SO"] > 0, agg_df["BB"] / agg_df["SO"], agg_df["BB"])
    agg_df["C%"] = np.where(agg_df["AB"] > 0, 1 - (agg_df["SO"] / agg_df["AB"]), 0)
    agg_df["HHB%"] = np.where(agg_df["AB"] > 0, agg_df["HHB"] / agg_df["AB"], 0)

    total_batted = agg_df["LD"] + agg_df["FB"] + agg_df["GB"]
    agg_df["LD%"] = np.where(total_batted > 0, agg_df["LD"] / total_batted, 0)
    agg_df["FB%"] = np.where(total_batted > 0, agg_df["FB"] / total_batted, 0)
    agg_df["GB%"] = np.where(total_batted > 0, agg_df["GB"] / total_batted, 0)

    denom = agg_df["AB"] - agg_df["SO"] - agg_df["HR"] + agg_df["SF"]
    agg_df["BABIP"] = np.where(denom > 0, (agg_df["H"] - agg_df["HR"]) / denom, 0)
    agg_df["BA/RISP"] = np.where(agg_df["AB_RISP"] > 0, agg_df["H_RISP"] / agg_df["AB_RISP"], 0)
    agg_df["PS/PA"] = np.where(agg_df["PA"] > 0, agg_df["PS"] / agg_df["PA"], 0)

    pct_cols = [
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "QAB%",
        "BB/K",
        "C%",
        "HHB%",
        "LD%",
        "FB%",
        "GB%",
        "BABIP",
        "BA/RISP",
        "PS/PA",
    ]
    agg_df[pct_cols] = agg_df[pct_cols].round(3)

    final_cols = [
        "Last",
        "First",
        "PA",
        "AB",
        "AVG",
        "OBP",
        "OPS",
        "SLG",
        "H",
        "R",
        "RBI",
        "BB",
        "2B",
        "SB",
        "QAB",
        "QAB%",
        "BB/K",
        "C%",
        "HHB",
        "HHB%",
        "LD%",
        "FB%",
        "GB%",
        "BABIP",
        "BA/RISP",
        "2OUTRBI",
        "XBH",
        "TB",
        "PS/PA",
        "SO",
    ]
    agg_df = agg_df[[c for c in final_cols if c in agg_df.columns]]
    return agg_df


# ============================
# Aggregation (Fielding)
# ============================

def aggregate_stats_fielding(csv_files):
    cols_to_keep = ["TC", "A", "PO", "E", "DP"]
    dfs = []
    for name in csv_files:
        file = f"{name}.csv"
        try:
            df = pd.read_csv(file, header=1)
            df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
            df.columns = [c.replace(".1", "") for c in df.columns]

            if "Last" not in df.columns:
                df["Last"] = ""
            if "First" not in df.columns:
                df["First"] = ""

            df = df[[c for c in cols_to_keep + ["Last", "First"] if c in df.columns]]

            df["Last"] = df["Last"].astype(str).str.strip().str.title()
            df["First"] = df["First"].astype(str).str.strip().str.title()

            for col in df.columns:
                if col not in ["Last", "First"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            dfs.append(df)
        except FileNotFoundError:
            continue
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame(columns=cols_to_keep + ["Last", "First", "Plays Made"])

    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)

    for col in agg_df.columns:
        if col not in ["Last", "First"]:
            agg_df[col] = agg_df[col].fillna(0).round(3)

    agg_df["FPCT"] = ((agg_df["A"] + agg_df["PO"]) / agg_df["TC"]).round(3).fillna(0)
    return agg_df


# ============================
# Aggregation (Catching)
# ============================

def aggregate_stats_catching(csv_files):
    cols_to_keep = ["INN", "PB", "SB", "SB-ATT", "CS"]
    dfs = []
    for name in csv_files:
        file = f"{name}.csv"
        try:
            df = pd.read_csv(file, header=1)
            df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
            df.columns = [c.replace(".1", "") for c in df.columns]
            df.columns = [c.replace(".2", "") for c in df.columns]

            if "Last" not in df.columns:
                df["Last"] = ""
            if "First" not in df.columns:
                df["First"] = ""

            df = df[[c for c in cols_to_keep + ["Last", "First"] if c in df.columns]]

            df["Last"] = df["Last"].astype(str).str.strip().str.title()
            df["First"] = df["First"].astype(str).str.strip().str.title()

            for col in df.columns:
                if col not in ["Last", "First", "SB-ATT"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            dfs.append(df)
        except FileNotFoundError:
            continue
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame(columns=cols_to_keep + ["Last", "First"])

    combined = pd.concat(dfs, ignore_index=True)

    if "SB-ATT" in combined.columns:
        split_sb_att = combined["SB-ATT"].astype(str).str.split("-", expand=True)
        if split_sb_att.shape[1] < 2:
            split_sb_att[1] = np.nan
        combined["SB"] = pd.to_numeric(split_sb_att[0], errors="coerce").fillna(0).astype(int)
        combined["ATT"] = pd.to_numeric(split_sb_att[1], errors="coerce").fillna(0).astype(int)
    else:
        combined["SB"] = 0
        combined["ATT"] = 0

    agg_df = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)

    for col in agg_df.columns:
        if col not in ["Last", "First", "INN"]:
            agg_df[col] = agg_df[col].fillna(0).round(0)

    agg_df["CS%"] = np.where(agg_df["ATT"] > 0, (agg_df["CS"] / agg_df["ATT"] * 100).round(1), 0)
    agg_df["SB-ATT"] = agg_df["SB"].astype(int).astype(str) + "-" + agg_df["ATT"].astype(int).astype(str)
    agg_df = agg_df.drop(columns=["SB", "ATT"])
    return agg_df


# =====================
# Acronym tables (from original)
# =====================

HITTING_KEY = pd.DataFrame({
    "Acronym": [
        "PA", "AB", "H", "AVG", "OBP", "SLG", "OPS", "RBI", "R",
        "BB", "SO", "XBH", "2B", "3B", "HR", "TB", "SB",
        "PS/PA", "BB/K", "C%",
        "QAB", "QAB%", "HHB", "HHB %",
        "LD%", "FB%", "GB%",
        "BABIP", "BA/RISP", "2OUTRBI"
    ],
    "Meaning": [
        "Plate Appearances", "At-Bats", "Hits", "Batting Average", "On-Base Percentage",
        "Slugging Percentage", "On-base Plus Slugging", "Runs Batted In", "Runs Scored",
        "Walks", "Strikeouts", "Extra-Base Hits", "Doubles", "Triples", "Home Runs",
        "Total Bases", "Stolen Bases", "Pitches per Plate Appearance", "Walk-to-Strikeout Ratio",
        "Contact Percentage", "Quality At-Bats", "Quality At-Bat Percentage", "Hard-Hit Balls",
        "Hard-Hit Ball Percentage", "Line Drive %", "Fly Ball %", "Ground Ball %",
        "Batting Average on Balls In Play", "Avg. w/ RISP", "Two-Out RBIs"
    ]
})

PITCHING_KEY = pd.DataFrame({
    "Acronym": [
        "IP", "ERA", "WHIP", "H", "R", "ER", "BB", "BB/INN", "SO", "K-L", "HR",
        "S%", "FPS%", "FPSO%", "FPSH%", "SM%", "<3%",
        "LD%", "FB%", "GB%", "HHB%", "WEAK%", "BBS",
        "BAA", "BABIP", "BA/RISP", "CS", "SB", "SB%", "FIP"
    ],
    "Meaning": [
        "Innings Pitched", "Earned Run Average", "Walks + Hits per Inning", "Hits Allowed",
        "Runs Allowed", "Earned Runs", "Walks", "Walks per Inning", "Strikeouts",
        "Strikeouts Looking", "Home Runs Allowed", "Strike %", "First-Pitch Strike %",
        "% of FPS ABs that end in outs", "% of FPS that are hits", "Swinging Miss %",
        "% of ABs with ≤3 pitches", "Line Drive %", "Fly Ball %", "Ground Ball %",
        "Hard-Hit Ball %", "Weak Contact %", "Base on Balls that results in a run",
        "Batting Avg Against", "BABIP", "Avg. w/ RISP", "Caught Stealing", "Stolen Bases Allowed",
        "Stolen Base %", "Fielding Independent Pitching"
    ]
})


# =====================
# Extra helpers to support requested UI
# =====================

def list_series_csvs():
    files = glob.glob("*.csv")
    # Exclude obvious cumulative names if present
    exclude = {"cumulative_hitting.csv", "cumulative_pitching.csv", "cumulative_fielding.csv", "cumulative_catching.csv"}
    return sorted({os.path.splitext(os.path.basename(f))[0] for f in files if os.path.basename(f) not in exclude})


def _drop_rows_nan_names(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Last" in df.columns and "First" in df.columns:
        mask = (~df["Last"].astype(str).str.strip().isin(["", "nan", "None"])) | (~df["First"].astype(str).str.strip().isin(["", "nan", "None"]))
        return df[mask].copy()
    return df


def _append_totals(df: pd.DataFrame, tab_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    num_cols = [c for c in df.columns if c not in ("Last", "First") and pd.api.types.is_numeric_dtype(df[c])]
    totals = df[num_cols].sum(numeric_only=True)
    total_row = {**{"Last": "TOTAL", "First": ""}, **{c: totals.get(c, 0) for c in num_cols}}
    # Recompute a few common rates for totals where feasible
    out = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return out


def _pitching_ip_gt_zero(df: pd.DataFrame) -> pd.DataFrame:
    if "IP" in df.columns:
        return df[df["IP"].fillna(0) > 0].copy()
    return df


def filter_players(df: pd.DataFrame, selected_players: list) -> pd.DataFrame:
    if not selected_players or "Last" not in df.columns:
        return df
    return df[df["Last"].isin(selected_players)].copy()


def extract_all_players(frames: dict) -> list:
    names = []
    for k, df in frames.items():
        if isinstance(df, pd.DataFrame) and not df.empty and "Last" in df.columns:
            names.extend(df["Last"].dropna().astype(str).unique().tolist())
    return sorted(set(names))


def _apply_display_formatting(df: pd.DataFrame, tab_name: str):
    # Reuse the rate formatting toggle defined below via RATE_COLS
    formatted = _fmt_rates(df, "Hitting" if tab_name == "Hitting" else ("Pitching" if tab_name == "Pitching" else ""))
    column_config = {}
    return formatted, column_config


# Provide Fielding/Catching keys to match UI
FIELDING_KEY = pd.DataFrame({
    "Acronym": ["TC", "A", "PO", "E", "DP", "FPCT"],
    "Meaning": [
        "Total Chances", "Assists", "Putouts", "Errors", "Double Plays", "Fielding Percentage"
    ],
})

CATCHING_KEY = pd.DataFrame({
    "Acronym": ["INN", "PB", "SB-ATT", "CS", "CS%"],
    "Meaning": [
        "Innings Caught", "Passed Balls", "Stolen Base Attempts (SB-ATT)", "Caught Stealing", "Caught Stealing %"
    ],
})


# Data loading helpers for the new UI

def load_series(stat_types: list, selected_series: list):
    frames = {t: pd.DataFrame() for t in STAT_TYPES_ALL}
    if not selected_series:
        return frames
    # Hitting
    if "Hitting" in stat_types:
        try:
            _hit = aggregate_stats_hitting(selected_series)
            frames["Hitting"] = prepare_batting_stats(_hit)
        except Exception:
            frames["Hitting"] = pd.DataFrame()
    # Pitching
    if "Pitching" in stat_types:
        try:
            _pit = aggregate_stats_pitching(selected_series)
            frames["Pitching"] = prepare_pitching_stats(generate_aggregated_pitching_df(_pit))
        except Exception:
            frames["Pitching"] = pd.DataFrame()
    # Fielding
    if "Fielding" in stat_types:
        try:
            _fld = aggregate_stats_fielding(selected_series)
            frames["Fielding"] = prepare_fielding_stats(_fld)
        except Exception:
            frames["Fielding"] = pd.DataFrame()
    # Catching
    if "Catching" in stat_types:
        try:
            _cat = aggregate_stats_catching(selected_series)
            frames["Catching"] = prepare_catching_stats(clean_df(_cat))
        except Exception:
            frames["Catching"] = pd.DataFrame()
    return frames


def load_cumulative()():
    """Load season-to-date from a single cumulative.csv and split into
    Hitting / Pitching / Fielding / Catching using the file's second header row.
    Falls back to aggregating series on disk if cumulative.csv is missing.
    """
    frames = {t: pd.DataFrame() for t in STAT_TYPES_ALL}
    cum_path = "cumulative.csv"

    if os.path.exists(cum_path):
        try:
            df = pd.read_csv(cum_path, header=1)
            # Normalize column names (strip whitespace) and drop all-NaN cols
            df.columns = [str(c).strip() for c in df.columns]
            df = df.dropna(axis=1, how="all")

            # Clean names early
            if {"Last", "First"}.issubset(df.columns):
                df["Last"] = df["Last"].astype(str).str.strip().str.title()
                df["First"] = df["First"].astype(str).str.strip().str.title()

            # Identify section boundaries
            # Hitting columns are from after [Number, Last, First] up to before first 'IP'
            if "IP" in df.columns:
                ip_idx = df.columns.get_loc("IP")
            else:
                ip_idx = len(df.columns)

            # Fielding start at 'TC' (if present)
            fld_start = df.columns.get_loc("TC") if "TC" in df.columns else None
            cat_start = df.columns.get_loc("INN") if "INN" in df.columns else None

            # ---- Hitting slice ----
            hit_cols = list(df.columns[:ip_idx])
            # Keep only expected hitting fields
            hit_keep = [c for c in hit_cols if c in set([
                "Last","First","PA","AB","H","AVG","OBP","SLG","OPS","RBI","R","BB","SO","XBH","2B","3B","HR","TB","SB",
                "PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI","SF","HBP","H_RISP","AB_RISP","PS","1B","GIDP","GITP","CI"
            ])]
            hit_df = df[[c for c in hit_keep if c in df.columns]].copy()
            # Derive missing fields if present
            if "R" not in hit_df.columns and "RBI" in df.columns:
                pass  # leave as-is
            frames["Hitting"] = prepare_batting_stats(clean_df(hit_df)) if not hit_df.empty else pd.DataFrame()

            # ---- Pitching slice ----
            if fld_start is not None:
                pit_cols = list(df.columns[ip_idx:fld_start])
            elif cat_start is not None:
                pit_cols = list(df.columns[ip_idx:cat_start])
            else:
                pit_cols = list(df.columns[ip_idx:])

            # Prefer pitching-specific stats; remove suffixes like .1
            def _unsuffix(name: str):
                return name.split(".")[0]
            pit_df = df[[c for c in pit_cols]].copy()
            pit_df = pit_df.rename(columns={c: _unsuffix(c) for c in pit_df.columns})
            # Ensure name columns included
            for nc in ("Last","First"):
                if nc in df.columns and nc not in pit_df.columns:
                    pit_df[nc] = df[nc]
            # Keep only downstream-needed columns
            pit_keep = list({
                "Last","First","IP","ER","H","BB","R","SO","K-L","HR","#P","BF","HBP",
                "FPS%","FPSO%","FPSW%","FPSH%","S%","SM%","LD%","FB%","GB%","BABIP","BA/RISP",
                "CS","SB","SB%","<3%","HHB%","WEAK%","BBS","ERA","WHIP","BB/INN","FIP","BAA"
            } & set(pit_df.columns))
            pit_df = pit_df[pit_keep + ([] if not {"Last","First"}.issubset(pit_df.columns) else [])]
            if not pit_df.empty:
                pit_df = generate_aggregated_pitching_df(pit_df)
                frames["Pitching"] = prepare_pitching_stats(pit_df)

            # ---- Fielding slice ----
            if fld_start is not None:
                fld_cols = [c for c in ["Last","First","TC","A","PO","FPCT","E","DP","TP"] if c in df.columns]
                fld_df = df[fld_cols].copy()
                frames["Fielding"] = prepare_fielding_stats(fld_df)

            # ---- Catching slice ----
            if cat_start is not None:
                cat_cols = [c for c in ["Last","First","INN","PB","SB-ATT","CS","CS%","SB","PIK","CI"] if c in df.columns]
                # Some exports have suffixes like SB.2, CS.2, PIK.2, CI.1
                alt_map = {c: c.split(".")[0] for c in df.columns if any(c.startswith(x+".") for x in ["SB","CS","PIK","CI"])}
                cat_df = df[[c for c in set(cat_cols) | set(alt_map.keys()) if c in df.columns]].copy()
                if alt_map:
                    cat_df = cat_df.rename(columns=alt_map)
                frames["Catching"] = prepare_catching_stats(clean_df(cat_df))
        except Exception as e:
            # Fallback to series aggregation if anything goes wrong
            series = list_series_csvs()
            frames = load_series(STAT_TYPES_ALL, series) if series else {t: pd.DataFrame() for t in STAT_TYPES_ALL}
    else:
        # Aggregate all series in current folder as season-to-date
        series = list_series_csvs()
        frames = load_series(STAT_TYPES_ALL, series) if series else {t: pd.DataFrame() for t in STAT_TYPES_ALL}
    return frames

st.title("EUCB Stats (Fall 2025)")

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
            help="Series correspond to CSV base names (e.g., wake, jmu, unc)."
        )

# Load per selection
if data_source == "Cumulative (default)":
    frames = load_cumulative()
    frames = filter_qualified_frames(frames, QUAL_MINS)
else:
    if not selected_series:
        st.warning("Select at least one series to view stats.")
        st.stop()
    frames = load_series(stat_types if stat_types else STAT_TYPES_ALL, selected_series)

all_player_lastnames = extract_all_players(frames)
selected_players = st.multiselect(
    "Filter by player (Last name); leave empty for All",
    options=all_player_lastnames,
    default=[],
)

# Keep the rate-format checkbox from earlier for continuity
format_rates = st.checkbox("Format rates as .xxx where applicable", value=False)

# Tabs
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
        # Append totals row prior to formatting
        df_filtered = _append_totals(df_filtered, tab_name)
        if selected_players and tab_name == "Pitching":
            df_before = len(df_filtered)
            df_filtered = _pitching_ip_gt_zero(df_filtered)
            if df_filtered.empty:
                st.warning(
                    "No **Pitching** rows match selected player(s) with > 0 IP." if df_before > 0 else "No **Pitching** rows match selected player(s)."
                )
                continue

        if df_filtered.empty:
            if selected_players:
                st.warning(f"No **{tab_name}** rows match selected player(s).")
            else:
                st.info(f"No data for **{tab_name}** with current filters.")
            continue

        df_display, column_config = _apply_display_formatting(df_filtered, tab_name)

        st.subheader(f"{tab_name} Stats")
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
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

st.caption("Tip: Place your series CSVs in the working directory. Optional cumulative files: cumulative_hitting.csv, cumulative_pitching.csv, cumulative_fielding.csv, cumulative_catching.csv")
