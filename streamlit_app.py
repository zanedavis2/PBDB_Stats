import os
import io
import glob
import numpy as np
import pandas as pd
import streamlit as st

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
    columns_to_keep = [
        "Last",
        "First",
        "IP",
        "ERA",
        "WHIP",
        "SO",
        "K-L",
        "H",
        "R",
        "ER",
        "BB",
        "BB/INN",
        "FIP",
        "S%",
        "FPS%",
        "FPSO%",
        "FPSH%",
        "BAA",
        "BBS",
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
    ]

    df = df.copy()

    for col in [
        "IP",
        "ER",
        "H",
        "BB",
        "SO",
        "K-L",
        "HR",
        "#P",
        "BF",
        "Strikes",
        "FirstPitchStrikes",
        "FPSO",
        "FPSH",
        "GroundBalls",
        "FlyBalls",
        "LineDrives",
        "HardHitBalls",
        "WeakContact",
        "Under3Pitches",
        "SwingMisses",
        "BBS",
        "CS",
        "SB",
    ]:
        if col not in df.columns:
            df[col] = 0

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

    if "BA/RISP" not in df.columns:
        df["BA/RISP"] = 0.000

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
# Streamlit UI
# =====================

st.set_page_config(page_title="PBDB Stats Viewer", layout="wide")
st.title("PBDB Filterable Stat Display")
st.caption("Series aggregation and cumulative uploads in one place.")

# ---- Data source selection ----
mode = st.sidebar.radio("Data source", ["Select series (CSV on disk)", "Upload cumulative CSVs"], index=0)

def parse_series(text):
    return [s.strip() for s in text.split(",") if s.strip()]

# Containers for dataframes
hitting_df = None
pitching_df = None
fielding_df = None
catching_df = None

if mode == "Select series (CSV on disk)":
    st.sidebar.write("**Enter series names that match `<name>.csv>` in this folder**")
    default_series = "wake,jmu"
    series_text = st.sidebar.text_input("Series (comma-separated)", value=default_series)
    series_selection = parse_series(series_text)

    if series_selection:
        with st.spinner("Aggregating stats from selected series ..."):
            # Hitting
            try:
                _hit = aggregate_stats_hitting(series_selection)
                hitting_df = generate_aggregated_hitting_df(_hit)
                hitting_df = prepare_batting_stats(hitting_df)
            except Exception as e:
                st.warning(f"Hitting aggregation failed: {e}")

            # Pitching
            try:
                _pit = aggregate_stats_pitching(series_selection)
                pitching_df = generate_aggregated_pitching_df(_pit)
                pitching_df = prepare_pitching_stats(pitching_df)
            except Exception as e:
                st.warning(f"Pitching aggregation failed: {e}")

            # Fielding
            try:
                _fld = aggregate_stats_fielding(series_selection)
                fielding_df = prepare_fielding_stats(_fld)
            except Exception as e:
                st.warning(f"Fielding aggregation failed: {e}")

            # Catching
            try:
                _cat = aggregate_stats_catching(series_selection)
                catching_df = prepare_catching_stats(clean_df(_cat))
            except Exception as e:
                st.warning(f"Catching aggregation failed: {e}")
else:
    st.sidebar.write("Upload your four cumulative CSVs (already combined across series)")
    up_hit = st.sidebar.file_uploader("Cumulative Hitting CSV", type=["csv"], key="hit")
    up_pit = st.sidebar.file_uploader("Cumulative Pitching CSV", type=["csv"], key="pit")
    up_fld = st.sidebar.file_uploader("Cumulative Fielding CSV", type=["csv"], key="fld")
    up_cat = st.sidebar.file_uploader("Cumulative Catching CSV", type=["csv"], key="cat")

    if up_hit:
        df = pd.read_csv(up_hit)
        hitting_df = prepare_batting_stats(clean_df(df))
    if up_pit:
        df = pd.read_csv(up_pit)
        pitching_df = prepare_pitching_stats(clean_df(df))
    if up_fld:
        df = pd.read_csv(up_fld)
        fielding_df = prepare_fielding_stats(clean_df(df))
    if up_cat:
        df = pd.read_csv(up_cat)
        catching_df = prepare_catching_stats(clean_df(df))

# ---- Filters ----
with st.sidebar.expander("Filters", expanded=True):
    min_pa = st.number_input("Min PA (Hitting)", min_value=0, value=0, step=1)
    min_ip = st.number_input("Min IP (Pitching)", min_value=0.0, value=0.0, step=0.1)
    min_tc = st.number_input("Min TC (Fielding)", min_value=0, value=0, step=1)
    min_inn = st.number_input("Min INN (Catching)", min_value=0.0, value=0.0, step=0.1)
    name_filter = st.text_input("Filter by last name contains (all tables)", value="")

    format_rates = st.checkbox("Format rates as .xxx where applicable", value=False)

# ---- Display tabs ----
T1, T2, T3, T4 = st.tabs(["Hitting", "Pitching", "Fielding", "Catching"])

# Utility: apply common filters

def _apply_common_filters(df, name_filter_substr):
    if df is None or df.empty:
        return df
    out = df.copy()
    if name_filter_substr:
        out = out[out["Last"].astype(str).str.contains(name_filter_substr, case=False, na=False)]
    return out

# Utility: format .xxx for rate-like columns
RATE_COLS = {
    "Hitting": ["AVG", "OBP", "SLG", "OPS", "QAB%", "BB/K", "C%", "HHB%", "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "PS/PA"],
    "Pitching": ["ERA", "WHIP", "BB/INN", "S%", "FPS%", "FPSO%", "FPSH%", "SM%", "LD%", "FB%", "GB%", "HHB%", "WEAK%", "BAA", "BABIP", "BA/RISP", "SB%", "<3%"],
}

def _fmt_rates(df, which):
    if not format_rates or df is None or df.empty:
        return df
    cols = [c for c in RATE_COLS.get(which, []) if c in df.columns]
    out = df.copy()
    for c in cols:
        try:
            out[c] = out[c].apply(lambda x: (f"{x:.3f}" if pd.notna(x) else "")).str.replace("0.", ".", regex=False)
        except Exception:
            pass
    return out

# ---- Hitting ----
with T1:
    if hitting_df is None:
        st.info("No hitting data yet.")
    else:
        hd = hitting_df.copy()
        if "PA" in hd.columns:
            hd = hd[hd["PA"] >= min_pa]
        hd = _apply_common_filters(hd, name_filter)
        hd = _fmt_rates(hd, "Hitting")
        st.dataframe(hd, use_container_width=True)
        st.download_button("Download Hitting CSV", data=hd.to_csv(index=False).encode("utf-8"), file_name="hitting.csv")
        with st.expander("Acronym key"):
            st.dataframe(HITTING_KEY, use_container_width=True)

# ---- Pitching ----
with T2:
    if pitching_df is None:
        st.info("No pitching data yet.")
    else:
        pdx = pitching_df.copy()
        if "IP" in pdx.columns:
            pdx = pdx[pdx["IP"] >= min_ip]
        pdx = _apply_common_filters(pdx, name_filter)
        pdx = _fmt_rates(pdx, "Pitching")
        st.dataframe(pdx, use_container_width=True)
        st.download_button("Download Pitching CSV", data=pdx.to_csv(index=False).encode("utf-8"), file_name="pitching.csv")
        with st.expander("Acronym key"):
            st.dataframe(PITCHING_KEY, use_container_width=True)

# ---- Fielding ----
with T3:
    if fielding_df is None:
        st.info("No fielding data yet.")
    else:
        fd = fielding_df.copy()
        if "TC" in fd.columns:
            fd = fd[fd["TC"] >= min_tc]
        fd = _apply_common_filters(fd, name_filter)
        st.dataframe(fd, use_container_width=True)
        st.download_button("Download Fielding CSV", data=fd.to_csv(index=False).encode("utf-8"), file_name="fielding.csv")

# ---- Catching ----
with T4:
    if catching_df is None:
        st.info("No catching data yet.")
    else:
        cd = catching_df.copy()
        if "INN" in cd.columns:
            cd = cd[cd["INN"] >= min_inn]
        cd = _apply_common_filters(cd, name_filter)
        st.dataframe(cd, use_container_width=True)
        st.download_button("Download Catching CSV", data=cd.to_csv(index=False).encode("utf-8"), file_name="catching.csv")

st.caption("Tip: In 'Select series' mode, ensure your CSV files are named exactly as '<series>.csv' in the working directory.")
