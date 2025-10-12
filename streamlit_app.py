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
