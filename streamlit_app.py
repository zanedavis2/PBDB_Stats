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
        # Normalize to strings
        df["Last"] = df["Last"].astype(str).str.strip()
        df["First"] = df["First"].astype(str).str.strip()

        # Treat any casing of 'nan'/'none' or blanks as missing
        def _norm_missing(s):
            s = s.astype(str).str.strip()
            lower = s.str.lower()
            s = s.mask(lower.isin(["", "nan", "none"]))
            return s
        df["Last"] = _norm_missing(df["Last"])
        df["First"] = _norm_missing(df["First"])

        # Drop totals/empty tail if present (both names missing)
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
    Also preserves internal count columns (prefixed with '_') so totals can be
    recomputed correctly in the UI.
    """
    # Keep a superset so we have denominators for weighted totals
    needed_counts = [
        "IP", "ER", "H", "BB", "R", "SO", "K-L", "HR", "#P", "BF", "HBP",
        "Strikes", "FirstPitchStrikes", "FPSO", "FPSH", "GroundBalls", "FlyBalls",
        "LineDrives", "HardHitBalls", "WeakContact", "Under3Pitches", "SwingMisses",
        "BBS", "CS", "SB"
    ]

    df = df.copy()
    for c in needed_counts:
        if c not in df.columns:
            df[c] = 0

    # Avoid /0
    df["IP"] = df["IP"].replace(0, np.nan)
    df["BF"] = df["BF"].replace(0, np.nan)
    df["#P"] = df["#P"].replace(0, np.nan)

    # Derived metrics
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

    df["SB%"] = np.where((df["SB"] + df["CS"]) > 0, (df["SB"] / (df["SB"] + df["CS"]) * 100).round(2), 0)
    df["BAA"] = np.where((df["BF"] - df["BB"] - df["HBP"]) > 0, (df["H"] / (df["BF"] - df["BB"] - df["HBP"])) .round(3), 0)
    df["BABIP"] = np.where((df["BF"] - df["SO"] - df["HR"] - df["BB"] - df["HBP"]) > 0, ((df["H"] - df["HR"]) / (df["BF"] - df["SO"] - df["HR"] - df["BB"] - df["HBP"])) .round(3), 0)

    if "BA/RISP" not in df.columns:
        df["BA/RISP"] = 0.000

    # Preserve internal counts for totals (prefixed with '_')
    internal_cols = {
        "_IP": "IP", "_ER": "ER", "_H": "H", "_BB": "BB", "_HR": "HR", "_SO": "SO",
        "_NP": "#P", "_BF": "BF", "_HBP": "HBP", "_STR": "Strikes", "_FPS": "FirstPitchStrikes",
        "_FPSO": "FPSO", "_FPSH": "FPSH", "_GB": "GroundBalls", "_FB": "FlyBalls",
        "_LD": "LineDrives", "_HHB": "HardHitBalls", "_WEAK": "WeakContact",
        "_U3": "Under3Pitches", "_SM": "SwingMisses", "_BBS": "BBS", "_CS": "CS", "_SB": "SB"
    }
    for new, old in internal_cols.items():
        df[new] = df[old].fillna(0)

    # Final column order (display + internal at the end)
    columns_to_keep = [
        "Last", "First", "IP", "ERA", "WHIP", "SO", "K-L", "H", "R", "ER", "BB",
        "BB/INN", "FIP", "S%", "FPS%", "FPSO%", "FPSH%", "BAA", "BBS", "SM%",
        "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "CS", "SB", "SB%", "<3%", "HHB%", "WEAK%"
    ] + list(internal_cols.keys())

    for col in columns_to_keep:
        if col not in df.columns:
            df[col] = 0

    df = df[columns_to_keep].copy()
    return df


# ============================
# Aggregation (Hitting)
# ============================

def aggregate_stats_hitting(csv_files):
    import numpy as np
    import pandas as pd

    cols_to_keep = [
        "Last","First","PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","2B","3B","HR",
        "SB","CS","QAB","HHB","LD%","FB%","GB%","H_RISP","AB_RISP","PS","2OUTRBI","XBH",
    ]

    def _pct_to_ratio(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return np.where(s > 1.0, s / 100.0, s)

    dfs = []
    for name in csv_files:
        file = f"{name}.csv"
        df = pd.read_csv(file, header=1)
        df = df[[c for c in cols_to_keep if c in df.columns]].copy()

        # Clean names
        df["Last"] = df["Last"].astype(str).str.strip().str.title()
        df["First"] = df["First"].astype(str).str.strip().str.title()

        # Convert all numeric columns
        for col in df.columns:
            if col not in ["Last", "First"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Convert % to ratio and compute LD/GB/FB counts = ratio * AB
        ld_ratio = _pct_to_ratio(df.get("LD%", 0))
        gb_ratio = _pct_to_ratio(df.get("GB%", 0))
        fb_ratio = _pct_to_ratio(df.get("FB%", 0))

        df["LD"] = np.rint(ld_ratio * df.get("AB", 0)).astype(int)
        df["GB"] = np.rint(gb_ratio * df.get("AB", 0)).astype(int)
        df["FB"] = np.rint(fb_ratio * df.get("AB", 0)).astype(int)

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    agg_df = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)

    for c in ["LD", "GB", "FB"]:
        if c in agg_df.columns:
            agg_df[c] = agg_df[c].astype(int)

    return agg_df

def generate_aggregated_hitting_df(df):
    cols_to_keep = [
        "Last", "First", "PA", "AB", "H", "BB", "HBP", "SF", "TB", "R", "RBI", "SO",
        "2B", "3B", "HR", "SB", "CS", "QAB", "HHB", "LD", "FB", "GB",
        "H_RISP", "AB_RISP", "PS", "2OUTRBI", "XBH",
    ]

    for col in cols_to_keep:
        if col not in df.columns:
            df[col] = 0
    df = df[cols_to_keep].copy()

    for col in df.columns:
        if col not in ["Last", "First"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    agg_df = df.groupby(["Last", "First"], as_index=False).sum()

    # === Derived statistics ===
    agg_df["AVG"] = np.where(agg_df["AB"] > 0, agg_df["H"] / agg_df["AB"], 0)
    agg_df["OBP"] = np.where(
        (agg_df["AB"] + agg_df["BB"] + agg_df["HBP"] + agg_df["SF"]) > 0,
        (agg_df["H"] + agg_df["BB"] + agg_df["HBP"]) /
        (agg_df["AB"] + agg_df["BB"] + agg_df["HBP"] + agg_df["SF"]),
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

    # === Format percentages ===
    pct_cols = ["AVG", "OBP", "SLG", "OPS", "QAB%", "BB/K", "C%", "HHB%", "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "PS/PA"]
    agg_df[pct_cols] = agg_df[pct_cols].round(3)

    # === Final columns for display ===
    final_cols = [
        "Last", "First", "PA", "AB", "AVG", "OBP", "OPS", "SLG", "H", "R", "RBI", "BB",
        "2B", "3B", "HR", "SB", "QAB", "QAB%", "BB/K", "C%", "HHB", "HHB%", "LD%", "FB%",
        "GB%", "BABIP", "BA/RISP", "2OUTRBI", "XBH", "TB", "PS/PA", "SO",
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
# Constants & lightweight helpers for new UI
# =====================
STAT_TYPES_ALL = ["Hitting", "Pitching", "Fielding", "Catching"]
QUAL_MINS = {"Hitting": 1, "Pitching": 0.1, "Fielding": 1, "Catching": 0.1}

FIELDING_KEY = pd.DataFrame({
    "Acronym": ["TC", "A", "PO", "FPCT", "E", "DP"],
    "Meaning": [
        "Total Chances", "Assists", "Putouts", "Fielding Percentage",
        "Errors", "Double Plays involvement"
    ]
})

CATCHING_KEY = pd.DataFrame({
    "Acronym": ["INN", "PB", "SB-ATT", "CS", "CS%"],
    "Meaning": [
        "Innings Caught", "Passed Balls", "Stolen Base Attempts",
        "Caught Stealing", "Caught Stealing %"
    ]
})

# Files expected locally (keep it simple per user's note)
CUMULATIVE_FILE = "cumulative.csv"

# Utility: list available series CSV base names (exclude cumulative)
def list_series_csvs():
    names = []
    for p in glob.glob("*.csv"):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.lower() != os.path.splitext(CUMULATIVE_FILE)[0].lower():
            names.append(base)
    # Stable order preference: alphabetical
    return sorted(names)

# Utility: drop rows with missing names
def _drop_rows_nan_names(df):
    if df is None or df.empty:
        return df
    # Coerce 'Nan', 'nan', 'NaN', 'None', '' to real NaN before dropping rows with both names missing
    for c in [col for col in ["Last", "First"] if col in df.columns]:
        s = df[c].astype(str).str.strip()
        df[c] = s.mask(s.str.lower().isin(["", "nan", "none"]))
    cols = [c for c in ["Last", "First"] if c in df.columns]
    if not cols:
        return df
    return df.dropna(subset=cols, how="all").reset_index(drop=True).reset_index(drop=True)

# Utility: append a totals row (sum numeric columns)
def _append_totals(df, tab_name):
    if df is None or df.empty:
        return df
    base = df.copy()

    # Helper to safe sum (robust if column missing)
    def ssum(col):
        if col in base.columns:
            return pd.to_numeric(base[col], errors="coerce").fillna(0).sum()
        return 0.0

    total_row = {c: "" for c in base.columns}
    total_row["Last"] = "Totals"
    total_row["First"] = ""

    if tab_name == "Pitching":
        # Sum raw counts
        IP = ssum("_IP") if "_IP" in base.columns else ssum("IP")
        ER = ssum("_ER") if "_ER" in base.columns else ssum("ER")
        H  = ssum("_H")  if "_H"  in base.columns else ssum("H")
        BB = ssum("_BB") if "_BB" in base.columns else ssum("BB")
        HR = ssum("_HR") if "_HR" in base.columns else ssum("HR")
        SO = ssum("_SO") if "_SO" in base.columns else ssum("SO")
        BF = ssum("_BF") if "_BF" in base.columns else ssum("BF")
        NP = ssum("_NP") if "_NP" in base.columns else ssum("#P")
        HBP= ssum("_HBP") if "_HBP" in base.columns else ssum("HBP")
        STR= ssum("_STR") if "_STR" in base.columns else np.nan
        FPS= ssum("_FPS") if "_FPS" in base.columns else np.nan
        FPSO= ssum("_FPSO") if "_FPSO" in base.columns else np.nan
        FPSH= ssum("_FPSH") if "_FPSH" in base.columns else np.nan
        GBc= ssum("_GB") if "_GB" in base.columns else np.nan
        FBc= ssum("_FB") if "_FB" in base.columns else np.nan
        LDc= ssum("_LD") if "_LD" in base.columns else np.nan
        HHBc= ssum("_HHB") if "_HHB" in base.columns else np.nan
        WEAKc= ssum("_WEAK") if "_WEAK" in base.columns else np.nan
        U3 = ssum("_U3") if "_U3" in base.columns else np.nan
        SMc= ssum("_SM") if "_SM" in base.columns else np.nan
        CS = ssum("_CS") if "_CS" in base.columns else ssum("CS")
        SB = ssum("_SB") if "_SB" in base.columns else ssum("SB")

        # Derived totals
        def rdiv(n, d):
            return (n / d) if (d and d != 0 and not np.isnan(d)) else np.nan

        total_row.update({
            "IP": round(IP if not np.isnan(IP) else 0, 2),
            "ER": ER, "H": H, "BB": BB, "HR": HR, "SO": SO,
            "ERA": round(rdiv(ER * 9, IP) or 0, 2),
            "WHIP": round(rdiv(BB + H, IP) or 0, 2),
            "BB/INN": round(rdiv(BB, IP) or 0, 2),
            "FIP": round((rdiv((13 * HR + 3 * BB - 2 * SO), IP) or 0) + 3.1, 2),
            "CS": CS, "SB": SB,
        })
        # Percentages if denominators exist
        if pd.notna(NP) and NP > 0 and pd.notna(STR):
            total_row["S%"] = round(STR / NP * 100, 2)
        if pd.notna(BF) and BF > 0:
            if pd.notna(FPS): total_row["FPS%"] = round(FPS / BF * 100, 2)
            if pd.notna(FPSO): total_row["FPSO%"] = round(FPSO / BF * 100, 2)
            if pd.notna(FPSH): total_row["FPSH%"] = round(FPSH / BF * 100, 2)
            if pd.notna(U3): total_row["<3%"] = round(U3 / BF * 100, 2)
        bb_balls = (BF - SO - BB - HBP) if pd.notna(BF) else np.nan
        if pd.notna(NP) and NP > 0 and pd.notna(SMc):
            total_row["SM%"] = round(SMc / NP * 100, 2)
        if pd.notna(bb_balls) and bb_balls > 0:
            if pd.notna(GBc): total_row["GB%"] = round(GBc / bb_balls * 100, 2)
            if pd.notna(FBc): total_row["FB%"] = round(FBc / bb_balls * 100, 2)
            if pd.notna(LDc): total_row["LD%"] = round(LDc / bb_balls * 100, 2)
            if pd.notna(HHBc): total_row["HHB%"] = round(HHBc / bb_balls * 100, 2)
            if pd.notna(WEAKc): total_row["WEAK%"] = round(WEAKc / bb_balls * 100, 2)
        if (SB + CS) > 0:
            total_row["SB%"] = round(SB / (SB + CS) * 100, 2)
        # BAA/BABIP
        denom_baa = (BF - BB - HBP) if pd.notna(BF) else np.nan
        if pd.notna(denom_baa) and denom_baa > 0:
            total_row["BAA"] = round(H / denom_baa, 3)
        denom_babip = (BF - SO - HR - BB - HBP) if pd.notna(BF) else np.nan
        if pd.notna(denom_babip) and denom_babip > 0:
            total_row["BABIP"] = round((H - HR) / denom_babip, 3)

    elif tab_name == "Hitting":
        # Sum raw counts
        for c in ["PA","AB","H","BB","HBP","SF","TB","R","RBI","SO","2B","3B","HR","SB","CS","QAB","HHB","LD","FB","GB","H_RISP","AB_RISP","PS","2OUTRBI","XBH"]:
            total_row[c] = ssum(c)
        # Derived
        AB = total_row.get("AB", 0)
        H = total_row.get("H", 0)
        BB = total_row.get("BB", 0)
        HBP = total_row.get("HBP", 0)
        SF = total_row.get("SF", 0)
        PA = total_row.get("PA", 0)
        TB = total_row.get("TB", 0)
        SO = total_row.get("SO", 0)
        HR = total_row.get("HR", 0)
        denom_obp = AB + BB + HBP + SF
        total_row.update({
            "AVG": round((H/AB) if AB else 0, 3),
            "OBP": round(((H+BB+HBP)/denom_obp) if denom_obp else 0, 3),
            "SLG": round((TB/AB) if AB else 0, 3),
            "OPS": 0,  # fill after SLG/OBP
            "QAB%": round((total_row["QAB"] / PA) if PA else 0, 3),
            "BB/K": round((BB/SO) if SO else BB, 3),
            "C%": round((1 - (SO/AB)) if AB else 0, 3),
        })
        total_row["OPS"] = round(total_row["OBP"] + total_row["SLG"], 3)
        total_batted = total_row["LD"] + total_row["FB"] + total_row["GB"]
        total_row["LD%"] = round((total_row["LD"]/total_batted) if total_batted else 0, 3)
        total_row["FB%"] = round((total_row["FB"]/total_batted) if total_batted else 0, 3)
        total_row["GB%"] = round((total_row["GB"]/total_batted) if total_batted else 0, 3)
        denom_babip = AB - SO - HR + SF
        total_row["BABIP"] = round(((H - HR)/denom_babip) if denom_babip else 0, 3)
        total_row["BA/RISP"] = round((total_row["H_RISP"]/total_row["AB_RISP"]) if total_row["AB_RISP"] else 0, 3)
        total_row["PS/PA"] = round((total_row["PS"] / PA) if PA else 0, 3)
        total_row["HHB%"] = round((total_row["HHB"]/AB) if AB else 0, 3)

    elif tab_name == "Fielding":
        for c in ["TC","A","PO","E","DP"]:
            total_row[c] = ssum(c)
        TC = total_row["TC"]
        total_row["FPCT"] = round(((total_row["A"] + total_row["PO"]) / TC) if TC else 0, 3)

    elif tab_name == "Catching":
        for c in ["INN","PB","CS"]:
            total_row[c] = ssum(c)
        # Handle SB-ATT split if present in dataframe
        if "SB-ATT" in base.columns:
            # base already split during aggregation; reconstruct totals from SB and ATT if available
            if "ATT" in base.columns:
                ATT = pd.to_numeric(base["ATT"], errors="coerce").fillna(0).sum()
            else:
                # fallback: parse each row's SB-ATT
                ATT = base["SB-ATT"].astype(str).str.split("-", expand=True).iloc[:,1].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
            SB = ssum("SB") if "SB" in base.columns else base["SB-ATT"].astype(str).str.split("-", expand=True).iloc[:,0].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
        else:
            SB = ssum("SB")
            ATT = SB + ssum("CS")
        total_row["SB-ATT"] = f"{int(SB)}-{int(ATT)}"
        total_row["CS%"] = round((total_row["CS"]/ATT * 100) if ATT else 0, 2)

    # Compose totals row and return
    totals_df = pd.DataFrame([total_row])
    return pd.concat([base, totals_df], ignore_index=True)

# Utility: filter selected players (by Last)
def filter_players(df, selected_lastnames):
    if not selected_lastnames:
        return df
    if "Last" not in df.columns:
        return df
    return df[df["Last"].isin(selected_lastnames)].copy()

# Pitching IP > 0 filter
def _pitching_ip_gt_zero(df):
    if "IP" not in df.columns:
        return df
    return df[df["IP"].fillna(0) > 0].copy()

# Extract all player last names from frames (dict of dfs)
def extract_all_players(frames):
    names = set()
    for df in frames.values():
        if df is not None and not df.empty and "Last" in df.columns:
            names.update(df["Last"].dropna().astype(str))
    return sorted(names)

# Display formatting (rates -> .xxx where relevant + Streamlit column_config)
RATE_COLS = {
    "Hitting": ["AVG", "OBP", "SLG", "OPS", "QAB%", "BB/K", "C%", "HHB%", "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "PS/PA"],
    "Pitching": ["ERA", "WHIP", "BB/INN", "S%", "FPS%", "FPSO%", "FPSH%", "SM%", "LD%", "FB%", "GB%", "HHB%", "WEAK%", "BAA", "BABIP", "BA/RISP", "SB%", "<3%"],
}

def _apply_display_formatting(df, tab_name):
    if df is None or df.empty:
        return df, {}
    out = df.copy()

    # Hide internal columns (prefixed with '_')
    internal_cols = [c for c in out.columns if isinstance(c, str) and c.startswith('_')]
    out = out[[c for c in out.columns if c not in internal_cols]]

    # Percent columns -> XX.XX%
    pct_cols = [c for c in out.columns if isinstance(c, str) and c.strip().endswith('%') and pd.api.types.is_numeric_dtype(out[c])]
    for c in pct_cols:
        s = out[c].astype(float)
        vals = s.replace([np.inf, -np.inf], np.nan).dropna().abs()
        if len(vals) == 0:
            scale = 1.0
        else:
            maxv = vals.max()
            scale = 100.0 if maxv <= 1.0 else (10.0 if maxv <= 10.0 else 1.0)
        out[c] = (s * scale).map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

    # Non-percent formatting
    if tab_name == "Pitching":
        # All remaining numeric (non-% columns) to 2 decimals
        for c in out.columns:
            if isinstance(c, str) and not c.endswith('%') and pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    else:
        # Hitting non-% rate-like columns as .xxx
        non_pct_rate_cols = [c for c in RATE_COLS.get(tab_name, []) if c in out.columns and not c.strip().endswith('%')]
        for c in non_pct_rate_cols:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].apply(lambda x: (f"{x:.3f}" if pd.notna(x) else "")).str.replace("0.", ".", regex=False)

    column_config = {}
    return out, column_config

# Data loading: cumulative -> read one file and let the prep funcs select columns

def load_cumulative():
    possible_paths = [
        "cumulative.csv",
        "/mnt/data/cumulative.csv",
    ]
    # Include any CSV that has 'cumulative' in the name (case-insensitive)
    possible_paths += [
        p for p in glob.glob("*.csv") + glob.glob("/mnt/data/*.csv")
        if "cumulative" in os.path.basename(p).lower()
    ]

    df_all = None
    for path in possible_paths:
        try:
            if os.path.exists(path):
                # ✅ Read with header=1 to skip the "Batting" row
                df_all = pd.read_csv(path, header=1)
                #st.success(f"Loaded cumulative file: {os.path.basename(path)}")
                break
        except Exception as e:
            st.warning(f"Failed reading {path}: {e}")

    if df_all is None or df_all.empty:
        st.error("No valid cumulative CSV found.")
        return {s: pd.DataFrame() for s in STAT_TYPES_ALL}

    # Clean and prepare the four stat sections
    df_all = clean_df(df_all)

    frames = {
        "Hitting": prepare_batting_stats(df_all),
        "Pitching": prepare_pitching_stats(df_all),
        "Fielding": prepare_fielding_stats(df_all),
        "Catching": prepare_catching_stats(df_all),
    }

    # Debugging info (optional)
    #st.write("✅ Columns detected:", list(df_all.columns)[:20])
    #st.write("✅ Rows detected:", len(df_all))

    return frames

def load_series(stat_types, selected_series):
    frames = {}
    sel = selected_series or []
    if "Hitting" in stat_types:
        try:
            _hit = aggregate_stats_hitting(sel)
            frames["Hitting"] = prepare_batting_stats(generate_aggregated_hitting_df(_hit))
        except Exception:
            frames["Hitting"] = pd.DataFrame()
    if "Pitching" in stat_types:
        try:
            _pit = aggregate_stats_pitching(sel)
            frames["Pitching"] = prepare_pitching_stats(generate_aggregated_pitching_df(_pit))
        except Exception:
            frames["Pitching"] = pd.DataFrame()
    if "Fielding" in stat_types:
        try:
            _fld = aggregate_stats_fielding(sel)
            frames["Fielding"] = prepare_fielding_stats(_fld)
        except Exception:
            frames["Fielding"] = pd.DataFrame()
    if "Catching" in stat_types:
        try:
            _cat = aggregate_stats_catching(sel)
            frames["Catching"] = prepare_catching_stats(clean_df(_cat))
        except Exception:
            frames["Catching"] = pd.DataFrame()
    return frames

# Apply qualification mins (drop rows below thresholds)

def filter_qualified_frames(frames, qual_mins):
    out = {}
    for key, df in frames.items():
        if df is None or df.empty:
            out[key] = df
            continue
        dfx = df.copy()
        if key == "Hitting" and "PA" in dfx.columns:
            dfx = dfx[dfx["PA"] >= qual_mins.get("Hitting", 0)]
        elif key == "Pitching" and "IP" in dfx.columns:
            dfx = dfx[dfx["IP"] >= qual_mins.get("Pitching", 0)]
        elif key == "Fielding" and "TC" in dfx.columns:
            dfx = dfx[dfx["TC"] >= qual_mins.get("Fielding", 0)]
        elif key == "Catching" and "INN" in dfx.columns:
            dfx = dfx[dfx["INN"] >= qual_mins.get("Catching", 0)]
        out[key] = dfx
    return out

# =====================
# NEW UI LAYOUT (same format/end product)
# =====================
st.set_page_config(page_title="EUCB Stats (Fall 2025)", layout="wide")

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

        df_display, column_config = _apply_display_formatting(df_filtered, tab_name)

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
