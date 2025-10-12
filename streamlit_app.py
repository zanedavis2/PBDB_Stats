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
