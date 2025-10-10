
import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# UI CONSTANTS
# =====================================================
STAT_TYPES_ALL = ["Hitting", "Pitching", "Fielding", "Catching"]
QUAL_MINS = {"Hitting": 1, "Pitching": 1, "Fielding": 1, "Catching": 1}

HITTING_KEY = pd.DataFrame({
    "Acronym": ["PA","AB","H","AVG","OBP","SLG","OPS","RBI","R","BB","SO","XBH","2B","3B","HR","TB","SB",
                "PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI"],
    "Meaning":  ["Plate Appearances","At-Bats","Hits","Batting Average","On-Base Percentage","Slugging Percentage",
                 "On-base Plus Slugging","Runs Batted In","Runs Scored","Walks","Strikeouts","Extra-Base Hits",
                 "Doubles","Triples","Home Runs","Total Bases","Stolen Bases","Pitches per Plate Appearance",
                 "Walk-to-Strikeout Ratio","Contact Percentage","Quality At-Bats","Quality At-Bat Percentage",
                 "Hard-Hit Balls","Hard-Hit Ball Percentage","Line Drive Percentage","Fly Ball Percentage",
                 "Ground Ball Percentage","Batting Average on Balls In Play",
                 "Batting Average with Runners in Scoring Position","Two-Out RBIs"]
})

PITCHING_KEY = pd.DataFrame({
    "Acronym": ["IP","ERA","WHIP","H","R","ER","BB","BB/INN","SO","K-L","HR","#P","BF","HBP",
                "S%","FPS%","FPSO%","FPSH%","SM%","<3%","LD%","FB%","GB%","HHB%","WEAK%","BBS",
                "BAA","BABIP","BA/RISP","CS","SB","SB%","FIP"],
    "Meaning":  ["Innings Pitched","Earned Run Average","Walks+Hits per Inning","Hits Allowed","Runs Allowed",
                 "Earned Runs","Walks","Walks per Inning","Strikeouts","Strikeouts Looking","Home Runs Allowed",
                 "Total Pitches","Batters Faced","Hit By Pitch",
                 "Strike Percentage","First-Pitch Strike %","% of FPS ABs that end in outs","% of FPS that are hits",
                 "Swinging Miss %","% of ABs with ≤3 pitches","Line Drive %","Fly Ball %","Ground Ball %",
                 "Hard-Hit Ball %","Weak Contact %","Base on Balls that score",
                 "Batting Average Against","Batting Average on Balls In Play","BA with RISP",
                 "Caught Stealing","Stolen Bases Allowed","Stolen Base %","Fielding Independent Pitching"]
})

FIELDING_KEY = pd.DataFrame({
    "Acronym": ["TC", "A", "PO", "E", "DP", "FPCT"],
    "Meaning": [
        "Total Chances","Assists","Putouts","Errors","Double Plays","Fielding Percentage"
    ]
})

CATCHING_KEY = pd.DataFrame({
    "Acronym": ["INN", "PB", "SB-ATT", "CS", "CS%"],
    "Meaning": [
        "Innings Caught","Passed Balls","Stolen Base Attempts Against","Runners Caught Stealing","Caught Stealing %"
    ]
})

# =====================================================
# HELPERS
# =====================================================
def _to_num(x):
    if isinstance(x, pd.DataFrame):
        return x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").fillna(0.0)
    try:
        return float(x)
    except Exception:
        return 0.0

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return _to_num(df[name])
    else:
        return pd.Series(0.0, index=df.index, dtype="float64")

def _safe_div(num, den) -> pd.Series:
    num = _to_num(num)
    den = _to_num(den)

    if isinstance(num, pd.Series) and not isinstance(den, pd.Series):
        den = pd.Series(float(den), index=num.index)
    if isinstance(den, pd.Series) and not isinstance(num, pd.Series):
        num = pd.Series(float(num), index=den.index)

    if isinstance(num, pd.Series) and isinstance(den, pd.Series):
        num, den = num.align(den, fill_value=0.0)
        out = pd.Series(0.0, index=num.index, dtype="float64")
        mask = den != 0
        out.loc[mask] = num.loc[mask] / den.loc[mask]
        return out

    return pd.Series(0.0 if float(den) == 0.0 else float(num) / float(den))

def _pct_to_ratio(series_or_val):
    """Accept percent in 0-1 or 0-100 scale, return as 0-1 float."""
    s = _to_num(series_or_val)
    if isinstance(s, pd.Series):
        return np.where(s > 1.5, s / 100.0, s)
    try:
        return s / 100.0 if float(s) > 1.5 else float(s)
    except Exception:
        return 0.0

def _convert_innings_value(ip):
    """Baseball tenths → decimal innings (4.1 -> 4 + 1/3, 4.2 -> 4 + 2/3)."""
    try:
        if pd.isna(ip):
            return np.nan
        ip = float(ip)
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

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace(".1", "").replace(".2", "").strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    if "Last" not in df.columns: df["Last"] = ""
    if "First" not in df.columns: df["First"] = ""

    # Drop rows where BOTH names are NaN-like
    df = _drop_rows_nan_names(df)

    # Title-case remaining names
    df["Last"] = df["Last"].astype(str).str.strip().str.title()
    df["First"] = df["First"].astype(str).str.strip().str.title()
    return df

def _smart_read_csv(path: str) -> pd.DataFrame:
    """
    Read a series CSV robustly:
    - Try header=1 (common for GameChanger exports), then header=0
    - If neither yields known columns, try to auto-detect the header row by scanning first 10 rows
    - Guarantee 'Last' and 'First' columns (rename first two text-like cols if necessary)
    """
    def _read_try(header):
        try:
            return pd.read_csv(path, header=header)
        except Exception:
            return None

    known_cols = {"Last", "First", "PA", "AB", "IP", "BF", "TC", "INN"}

    # Try common headers
    for h in (1, 0):
        df = _read_try(h)
        if df is not None and any(c in df.columns for c in known_cols):
            return df

    # Try to detect header row
    try:
        raw = pd.read_csv(path, header=None, nrows=20)
        header_row = None
        for i in range(min(10, len(raw))):
            row_vals = raw.iloc[i].astype(str).str.strip().tolist()
            if any(v in ("Last","First","PA","AB","IP","BF","TC","INN") for v in row_vals):
                header_row = i
                break
        if header_row is not None:
            df = pd.read_csv(path, header=header_row)
            return df
    except Exception:
        pass

    # Fallback
    df = pd.read_csv(path, header=0)
    return df

def _ensure_name_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    if "Last" not in cols and len(cols) >= 1:
        df.rename(columns={cols[0]: "Last"}, inplace=True)
    if "First" not in df.columns and len(df.columns) >= 2:
        df.rename(columns={df.columns[1]: "First"}, inplace=True)
    if "Last" not in df.columns:
        df["Last"] = ""
    if "First" not in df.columns:
        df["First"] = ""
    return df

def _apply_display_formatting(df: pd.DataFrame, tab_name: str):
    df_disp = df.copy()
    col_config = {}

    pct_cols = [c for c in df_disp.columns if isinstance(c, str) and c.endswith("%")]
    for c in pct_cols:
        # underlying numeric as 0-1; render as XX.X%
        df_disp[c] = _to_num(df_disp[c]) * 100.0
        col_config[c] = st.column_config.NumberColumn(format="%.1f%%")

    if tab_name == "Hitting":
        def _dot3(x):
            if pd.isna(x): return ""
            try:
                s = f"{float(x):.3f}"
            except Exception:
                return ""
            return s[1:] if s.startswith("0") else s
        for c in [c for c in ["AVG","OBP","SLG","OPS","BABIP"] if c in df_disp.columns]:
            df_disp[c] = _to_num(df_disp[c]).apply(_dot3)
            col_config[c] = st.column_config.TextColumn()

    return df_disp, col_config

def _pitching_ip_gt_zero(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "IP" not in df.columns:
        return df
    ip_dec = pd.to_numeric(df["IP"].apply(_convert_innings_value), errors="coerce").fillna(0.0)
    return df[ip_dec > 0.0].reset_index(drop=True)



def _is_nanlike_series(s: pd.Series) -> pd.Series:
    """Return boolean Series where entries are NaN-ish: NaN, empty, or strings like 'nan', 'none', 'null' (case-insensitive)."""
    s_str = s.astype(str).str.strip()
    return (
        s.isna()
        | s_str.eq("")
        | s_str.str.lower().isin({"nan", "none", "null", "n/a", "na"})
    )

def _drop_rows_nan_names(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Last","First"}.issubset(df.columns):
        return df
    last_raw = df["Last"]
    first_raw = df["First"]
    last_is_nan = _is_nanlike_series(last_raw)
    first_is_nan = _is_nanlike_series(first_raw)
    # drop when BOTH are nan-like
    keep = ~(last_is_nan & first_is_nan)
    return df.loc[keep].copy()

# =====================================================
# HITTING
# =====================================================
HIT_INPUT_COLS = [
    "Last","First","PA","AB","H","BB","HBP","SF","TB","R","RBI","SO",
    "2B","3B","HR","SB","CS","QAB","HHB","LD","FB","GB","H_RISP","AB_RISP",
    "PS","2 out RBI","XBH","QAB%","HHB%","LD%","FB%","GB%","C%"
]

def _backfill_hitting_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Coerce known numeric cols
    for c in [col for col in df.columns if col not in ("Last","First")]:
        df[c] = _to_num(df[c])

    # Balls in play (for LD/FB/GB)
    AB  = _col(df, "AB")
    SO  = _col(df, "SO")
    HR  = _col(df, "HR")
    SF  = _col(df, "SF")
    BIP = AB - SO - HR + SF
    BIP = BIP.clip(lower=0)

    # Reconstruct LD/FB/GB from % if counts absent or all zero
    for stat in ["LD","FB","GB"]:
        pct_col = f"{stat}%"
        if stat not in df.columns or _to_num(df[stat]).sum() == 0:
            if pct_col in df.columns:
                ratio = _pct_to_ratio(df[pct_col])
                df[stat] = np.rint(ratio * BIP).astype(float)

    # Reconstruct QAB from QAB% * PA if missing
    if ("QAB" not in df.columns or _to_num(df["QAB"]).sum() == 0) and "QAB%" in df.columns:
        df["QAB"] = np.rint(_pct_to_ratio(df["QAB%"]) * _col(df, "PA")).astype(float)

    # Reconstruct HHB from HHB% * AB (fallback to PA)
    if ("HHB" not in df.columns or _to_num(df["HHB"]).sum() == 0) and "HHB%" in df.columns:
        base = _col(df, "AB")
        base = np.where(base > 0, base, _col(df, "PA"))
        df["HHB"] = np.rint(_pct_to_ratio(df["HHB%"]) * base).astype(float)

    return df

def prepare_batting_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(df)
    keep = [c for c in HIT_INPUT_COLS if c in df.columns]
    df = df[["Last","First"] + [c for c in keep if c not in ("Last","First")]].copy()
    df = _backfill_hitting_counts(df)
    # Ensure minimal columns for grouping
    for _colname in ("Last","First","PA","AB"):
        if _colname not in df.columns:
            df[_colname] = 0

    # Numerics
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])

    # Derived rates
    PA  = _col(df, "PA")
    AB  = _col(df, "AB")
    H   = _col(df, "H")
    BB  = _col(df, "BB")
    HBP = _col(df, "HBP")
    SF  = _col(df, "SF")
    TB  = _col(df, "TB")
    SO  = _col(df, "SO")
    HR  = _col(df, "HR")
    QAB = _col(df, "QAB")
    HHB = _col(df, "HHB")
    LD  = _col(df, "LD")
    FB  = _col(df, "FB")
    GB  = _col(df, "GB")
    H_RISP  = _col(df, "H_RISP")
    AB_RISP = _col(df, "AB_RISP")
    PS = _col(df, "PS")

    batted = LD + FB + GB
    df["AVG"] = _safe_div(H, AB).round(3)
    df["OBP"] = _safe_div(H + BB + HBP, AB + BB + HBP + SF).round(3)
    df["SLG"] = _safe_div(TB, AB).round(3)
    df["OPS"] = (df["OBP"] + df["SLG"]).round(3)
    df["QAB%"] = _safe_div(QAB, PA).round(3)
    df["HHB%"] = _safe_div(HHB, AB.where(AB>0, PA)).round(3)
    df["LD%"] = _safe_div(LD, batted).round(3)
    df["FB%"] = _safe_div(FB, batted).round(3)
    df["GB%"] = _safe_div(GB, batted).round(3)
    df["BABIP"] = _safe_div(H - HR, AB - SO - HR + SF).round(3)
    df["BA/RISP"] = _safe_div(H_RISP, AB_RISP).round(3)
    df["PS/PA"] = _safe_div(PS, PA).round(3)
    df["BB/K"] = _safe_div(BB, SO.replace(0, np.nan)).fillna(0.0).round(3)
    df["C%"] = _safe_div(AB - SO, AB).round(3)

    # Friendly rename
    if "2 out RBI" in df.columns:
        df.rename(columns={"2 out RBI": "2OUTRBI"}, inplace=True)

    display_cols = [
        "Last","First","PA","AB","AVG","OBP","OPS","SLG","H","R","RBI","BB","SO","XBH","2B","3B","HR",
        "TB","SB","PS/PA","BB/K","C%","QAB","QAB%","HHB","HHB%","LD%","FB%","GB%","BABIP","BA/RISP","2OUTRBI"
    ]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing].copy()

def aggregate_stats_hitting(series_names):
    dfs = []
    for name in series_names:
        file = f"{name}.csv"
        if not os.path.exists(file):
            continue
        df = _ensure_name_cols(_smart_read_csv(file))
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=HIT_INPUT_COLS)
    combined = pd.concat(dfs, ignore_index=True)
    return _drop_rows_nan_names(clean_df(combined))

def generate_aggregated_hitting_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(df_raw)
    df = _backfill_hitting_counts(df)
    # Ensure minimal columns for grouping
    for _colname in ("Last","First","PA","AB"):
        if _colname not in df.columns:
            df[_colname] = 0
    # numeric
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])
    # aggregate by player (sum counts)
    agg = df.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    # recompute rates on combined (fresh averages)
    agg = _drop_rows_nan_names(clean_df(agg))
    return prepare_batting_stats(agg)

# =====================================================
# PITCHING
# =====================================================
PIT_INPUT_COLS = [
    "Last","First","IP","ER","H","R","BB","SO","K-L","HR","#P","BF","HBP",
    "S%","FPS%","FPSO%","FPSH%","SM%","LD%","FB%","GB%","BABIP","BA/RISP",
    "CS","SB","SB%","<3%","HHB%","WEAK%","BBS","ERA","WHIP","FIP","BB/INN","BAA"
]

def _backfill_pitching_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in [col for col in df.columns if col not in ("Last","First")]:
        df[c] = _to_num(df[c])

    # Decimal innings for calculations
    if "IP" in df.columns:
        df["_IP_DEC"] = df["IP"].apply(_convert_innings_value)
    else:
        df["_IP_DEC"] = 0.0

    # Batted balls = BF - SO - BB - HBP
    BF  = _col(df, "BF")
    SO  = _col(df, "SO")
    BB  = _col(df, "BB")
    HBP = _col(df, "HBP")
    BIP = (BF - SO - BB - HBP).clip(lower=0)

    # From % → counts (robust to scale 0-1 or 0-100)
    if "S%" in df.columns:
        df["Strikes"] = np.rint(_pct_to_ratio(df["S%"]) * _col(df, "#P")).astype(float)
    if "FPS%" in df.columns:
        df["FirstPitchStrikes"] = np.rint(_pct_to_ratio(df["FPS%"]) * BF).astype(float)
    if "FPSO%" in df.columns:
        df["FPSO"] = np.rint(_pct_to_ratio(df["FPSO%"]) * BF).astype(float)
    if "FPSH%" in df.columns:
        df["FPSH"] = np.rint(_pct_to_ratio(df["FPSH%"]) * BF).astype(float)
    if "SM%" in df.columns:
        df["SwingMisses"] = np.rint(_pct_to_ratio(df["SM%"]) * _col(df, "#P")).astype(float)
    if "LD%" in df.columns:
        df["LineDrives"] = np.rint(_pct_to_ratio(df["LD%"]) * BIP).astype(float)
    if "FB%" in df.columns:
        df["FlyBalls"] = np.rint(_pct_to_ratio(df["FB%"]) * BIP).astype(float)
    if "GB%" in df.columns:
        df["GroundBalls"] = np.rint(_pct_to_ratio(df["GB%"]) * BIP).astype(float)
    if "HHB%" in df.columns:
        df["HardHitBalls"] = np.rint(_pct_to_ratio(df["HHB%"]) * BIP).astype(float)
    if "WEAK%" in df.columns:
        df["WeakContact"] = np.rint(_pct_to_ratio(df["WEAK%"]) * BIP).astype(float)
    if "<3%" in df.columns:
        df["Under3Pitches"] = np.rint(_pct_to_ratio(df["<3%"]) * BF).astype(float)

    return df

def prepare_pitching_stats(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.columns) > 140:
        df = df.iloc[:, [0, 1] + list(range(53, 148))]
    df = clean_df(df)
    keep = [c for c in PIT_INPUT_COLS if c in df.columns]
    df = df[["Last","First"] + [c for c in keep if c not in ("Last","First")]].copy()
    df = _backfill_pitching_counts(df)
    for _colname in ("Last","First","IP","BF"):
        if _colname not in df.columns:
            df[_colname] = 0

    # Coerce numeric
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])

    # Decimal IP
    df["_IP_DEC"] = df["IP"].apply(_convert_innings_value) if "IP" in df.columns else 0.0

    # Derive core if missing / recompute from counts
    ER = _col(df, "ER"); H = _col(df, "H"); BB = _col(df, "BB"); SO = _col(df, "SO")
    HR = _col(df, "HR"); P = _col(df, "#P"); BF = _col(df, "BF"); HBP = _col(df, "HBP")
    Strikes = _col(df, "Strikes"); FPS = _col(df, "FirstPitchStrikes")
    FPSO = _col(df, "FPSO"); FPSH = _col(df, "FPSH")
    GB = _col(df, "GroundBalls"); FB = _col(df, "FlyBalls"); LD = _col(df, "LineDrives")
    HHB = _col(df, "HardHitBalls"); WEAK = _col(df, "WeakContact")
    U3 = _col(df, "Under3Pitches"); SM = _col(df, "SwingMisses")

    ip_dec = _col(df, "_IP_DEC")
    df["ERA"] = _safe_div(ER * 9.0, ip_dec).round(3)
    df["WHIP"] = _safe_div(BB + H, ip_dec).round(3)
    df["BB/INN"] = _safe_div(BB, ip_dec).round(3)
    C = 3.1
    df["FIP"] = (_safe_div(13*HR + 3*(BB + HBP) - 2*SO, ip_dec) + C).round(3)

    # Recompute percentages from counts
    BIP = (BF - SO - BB - HBP).clip(lower=0)
    df["S%"] = _safe_div(Strikes, P).round(3)
    df["FPS%"] = _safe_div(FPS, BF).round(3)
    df["FPSO%"] = _safe_div(FPSO, BF).round(3)
    df["FPSH%"] = _safe_div(FPSH, BF).round(3)
    df["SM%"] = _safe_div(SM, P).round(3)
    df["LD%"] = _safe_div(LD, BIP).round(3)
    df["FB%"] = _safe_div(FB, BIP).round(3)
    df["GB%"] = _safe_div(GB, BIP).round(3)
    df["HHB%"] = _safe_div(HHB, BIP).round(3)
    df["WEAK%"] = _safe_div(WEAK, BIP).round(3)
    df["<3%"] = _safe_div(U3, BF).round(3)

    # SB%
    SB = _col(df, "SB"); CS = _col(df, "CS")
    df["SB%"] = np.where((SB + CS) > 0, (SB / (SB + CS)).round(3), 0.0)

    # BAA, BABIP
    df["BAA"] = _safe_div(H, BF - BB - HBP).round(3)
    df["BABIP"] = _safe_div(H - HR, BF - SO - HR - BB - HBP).round(3)

    # Keep the display order
    display_cols = [
        "Last","First","IP","ERA","WHIP","SO","K-L","H","R","ER","BB","BB/INN",
        "FIP","S%","FPS%","FPSO%","FPSH%","BAA","BBS","SM%","LD%","FB%","GB%","BABIP",
        "BA/RISP","CS","SB","SB%","<3%","HHB%","WEAK%","#P","BF","HBP"
    ]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing].copy()

def aggregate_stats_pitching(series_names):
    dfs = []
    for name in series_names:
        file = f"{name}.csv"
        if not os.path.exists(file):
            continue
        df = _ensure_name_cols(_smart_read_csv(file))
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=PIT_INPUT_COLS)
    combined = pd.concat(dfs, ignore_index=True)
    return _drop_rows_nan_names(clean_df(combined))

def generate_aggregated_pitching_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = df_raw.copy()
    df_raw = df_raw.iloc[:, [0, 1] + list(range(53, 148))]
    df = clean_df(df_raw)
    df = _backfill_pitching_counts(df)
    for _colname in ("Last","First","IP","BF"):
        if _colname not in df.columns:
            df[_colname] = 0
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])
    agg = df.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
    agg = _drop_rows_nan_names(clean_df(agg))
    return prepare_pitching_stats(agg)

# =====================================================
# FIELDING
# =====================================================
FLD_INPUT_COLS = ["Last","First","TC","A","PO","FPCT","E","DP"]

def prepare_fielding_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(df)
    keep = [c for c in FLD_INPUT_COLS if c in df.columns]
    df = df[["Last","First"] + [c for c in keep if c not in ("Last","First")]].copy()
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])
    if "FPCT" not in df.columns and {"PO","A","TC"}.issubset(df.columns):
        df["FPCT"] = _safe_div(_col(df, "PO") + _col(df, "A"), _col(df, "TC")).round(3)
    display_cols = ["Last","First","TC","PO","A","E","DP","FPCT"]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing].copy()

def aggregate_stats_fielding(series_names):
    dfs = []
    for name in series_names:
        file = f"{name}.csv"
        if not os.path.exists(file):
            continue
        df = _ensure_name_cols(_smart_read_csv(file))
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=FLD_INPUT_COLS)
    combined = pd.concat(dfs, ignore_index=True)
    return _drop_rows_nan_names(clean_df(combined))

# =====================================================
# CATCHING
# =====================================================
CAT_INPUT_COLS = ["Last","First","INN","PB","SB-ATT","CS","CS%"]

def prepare_catching_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(df)
    keep = [c for c in CAT_INPUT_COLS if c in df.columns]
    df = df[["Last","First"] + [c for c in keep if c not in ("Last","First")]].copy()

    # --- Parse SB-ATT before numeric coercion ---
    if "SB-ATT" in df.columns:
        # Build numeric SB allowed and ATT from strings like "5-5"
        parsed = df["SB-ATT"].apply(_parse_sb_att)
        df["_SB_ALLOWED"] = parsed.apply(lambda t: t[0])
        df["_ATTEMPTS"]   = parsed.apply(lambda t: t[1])
        # Keep display column as attempts
        df["SB-ATT"] = df["_ATTEMPTS"]

        # If CS missing or zero, compute CS = ATTEMPTS - SB_ALLOWED
        if "CS" not in df.columns:
            df["CS"] = df["_ATTEMPTS"] - df["_SB_ALLOWED"]
        else:
            cs_num = pd.to_numeric(df["CS"], errors="coerce").fillna(0.0)
            need = cs_num == 0
            df.loc[need, "CS"] = (df.loc[need, "_ATTEMPTS"] - df.loc[need, "_SB_ALLOWED"]).astype(float)

    # Now coerce numerics
    for c in df.columns:
        if c not in ("Last","First"):
            df[c] = _to_num(df[c])

    # Compute CS% if missing or zero
    if "CS%" not in df.columns:
        df["CS%"] = 0.0
    att = df["SB-ATT"] if "SB-ATT" in df.columns else 0.0
    cs = df["CS"] if "CS" in df.columns else 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        cs_pct = np.where(_to_num(att) > 0, _to_num(cs) / _to_num(att), 0.0)
    df["CS%"] = cs_pct.round(3)

    # Clean up helper cols if they exist
    for helper in ["_SB_ALLOWED","_ATTEMPTS"]:
        if helper in df.columns:
            df.drop(columns=[helper], inplace=True)

    display_cols = ["Last","First","INN","PB","SB-ATT","CS","CS%"]
    existing = [c for c in display_cols if c in df.columns]
    return df[existing].copy()

def aggregate_stats_catching(series_names):
    dfs = []
    for name in series_names:
        file = f"{name}.csv"
        if not os.path.exists(file):
            continue
        df = _ensure_name_cols(_smart_read_csv(file))
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=CAT_INPUT_COLS)
    combined = pd.concat(dfs, ignore_index=True)
    return _drop_rows_nan_names(clean_df(combined))

# =====================================================
# DATA LOADING / ORCHESTRATION
# =====================================================
@st.cache_data(show_spinner=False)
def list_series_csvs():
    files = [os.path.basename(p) for p in glob.glob("*.csv") if not os.path.basename(p).startswith("~")]
    series = []
    for f in files:
        name = os.path.splitext(f)[0]
        if name.lower() != "cumulative":
            series.append(name)
    return sorted(series)

@st.cache_data(show_spinner=False)
def load_cumulative():
    if not os.path.exists("cumulative.csv"):
        return {"Hitting": pd.DataFrame(),"Pitching": pd.DataFrame(),
                "Fielding": pd.DataFrame(),"Catching": pd.DataFrame()}
    raw = pd.read_csv("cumulative.csv", header=1)
    raw = clean_df(raw)
    return {
        "Hitting":  prepare_batting_stats(raw.copy()),
        "Pitching": prepare_pitching_stats(raw.copy()),
        "Fielding": prepare_fielding_stats(raw.copy()),
        "Catching": prepare_catching_stats(raw.copy()),
    }

@st.cache_data(show_spinner=False)
def load_series(stat_types, series_names):
    out = {}
    if "Hitting" in stat_types:
        agg = aggregate_stats_hitting(series_names)
        out["Hitting"] = generate_aggregated_hitting_df(agg)
    if "Pitching" in stat_types:
        agg = aggregate_stats_pitching(series_names)
        out["Pitching"] = generate_aggregated_pitching_df(agg)
    if "Fielding" in stat_types:
        agg = aggregate_stats_fielding(series_names)
        for c in agg.columns:
            if c not in ("Last","First"):
                agg[c] = _to_num(agg[c])
        agg = agg.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
        out["Fielding"] = prepare_fielding_stats(agg)
    if "Catching" in stat_types:
        agg = aggregate_stats_catching(series_names)
        for c in agg.columns:
            if c not in ("Last","First"):
                agg[c] = _to_num(agg[c])
        agg = agg.groupby(["Last","First"], as_index=False).sum(numeric_only=True)
        out["Catching"] = prepare_catching_stats(agg)
    return out

def filter_players(df, selected_players):
    if df.empty or not selected_players:
        return df
    if "Last" not in df.columns:
        return df
    return df[df["Last"].isin(selected_players)].reset_index(drop=True)

def extract_all_players(frames_dict):
    names = set()
    for df in frames_dict.values():
        if isinstance(df, pd.DataFrame) and {"Last","First"}.issubset(df.columns):
            names.update(df["Last"].dropna().astype(str).tolist())
    return sorted(names)

def filter_qualified_frames(frames: dict, mins: dict) -> dict:
    out = {}
    df = frames.get("Hitting", pd.DataFrame()).copy()
    if not df.empty and "PA" in df.columns:
        pa = pd.to_numeric(df["PA"], errors="coerce").fillna(0)
        out["Hitting"] = df[pa >= int(mins.get("Hitting", 1))].reset_index(drop=True)
    else:
        out["Hitting"] = df
    df = frames.get("Pitching", pd.DataFrame()).copy()
    if not df.empty and "BF" in df.columns:
        bf = pd.to_numeric(df["BF"], errors="coerce").fillna(0)
        out["Pitching"] = df[bf >= int(mins.get("Pitching", 1))].reset_index(drop=True)
    else:
        out["Pitching"] = df
    df = frames.get("Fielding", pd.DataFrame()).copy()
    if not df.empty and "TC" in df.columns:
        tc = pd.to_numeric(df["TC"], errors="coerce").fillna(0)
        out["Fielding"] = df[tc >= int(mins.get("Fielding", 1))].reset_index(drop=True)
    else:
        out["Fielding"] = df
    df = frames.get("Catching", pd.DataFrame()).copy()
    if not df.empty and "INN" in df.columns:
        inn = pd.to_numeric(df["INN"], errors="coerce").fillna(0)
        out["Catching"] = df[inn >= float(mins.get("Catching", 1))].reset_index(drop=True)
    else:
        out["Catching"] = df
    return out



# =====================================================
# TOTALS ROW HELPERS

def _parse_sb_att(value):
    """
    Parse catcher 'SB-ATT' strings like '5-5' (SB-ATT) into (sb_allowed, attempts).
    Returns (sb_allowed, attempts) as floats. Handles None/NaN/empty gracefully.
    Accepts dashes '-', '–', '—', with/without spaces.
    """
    try:
        if pd.isna(value):
            return 0.0, 0.0
        s = str(value).strip()
        if not s:
            return 0.0, 0.0
        s = s.replace("—", "-").replace("–", "-")
        if "-" in s:
            parts = [p.strip() for p in s.split("-", 1)]
            if len(parts) == 2:
                sb = float(pd.to_numeric(parts[0], errors="coerce"))
                att = float(pd.to_numeric(parts[1], errors="coerce"))
                sb = 0.0 if pd.isna(sb) else sb
                att = 0.0 if pd.isna(att) else att
                return sb, att
        # If it's a single number, treat as attempts, 0 SB allowed
        att = float(pd.to_numeric(s, errors="coerce"))
        att = 0.0 if pd.isna(att) else att
        return 0.0, att
    except Exception:
        return 0.0, 0.0

# =====================================================
def _dec_to_baseball_tenths(ip_dec: float) -> float:
    """
    Convert decimal innings back to baseball-tenths (e.g., 4 + 1/3 -> 4.1, 4 + 2/3 -> 4.2).
    Returns a float like 7.2
    """
    try:
        whole = int(np.floor(ip_dec))
        frac = ip_dec - whole
        # closest of 0, 1/3, 2/3
        if abs(frac - 1/3) < 1e-6:
            return whole + 0.1
        elif abs(frac - 2/3) < 1e-6:
            return whole + 0.2
        else:
            # round to nearest third
            thirds = round(frac * 3)
            if thirds == 1:
                return whole + 0.1
            elif thirds == 2:
                return whole + 0.2
            else:
                return float(whole)
    except Exception:
        return float(ip_dec)

def _append_totals_row_hitting(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()

    # Convert numeric-like cols
    for c in tmp.columns:
        if c not in ("Last","First"):
            tmp[c] = _to_num(tmp[c])

    # Sum raw counts where available
    get = lambda c: _to_num(tmp[c]).sum() if c in tmp.columns else 0.0
    PA  = get("PA")
    AB  = get("AB")
    H   = get("H")
    BB  = get("BB")
    HBP = get("HBP")
    SF  = get("SF")
    TB  = get("TB")
    R   = get("R")
    RBI = get("RBI")
    SO  = get("SO")
    XBH = get("XBH")
    _2B = get("2B")
    _3B = get("3B")
    HR  = get("HR")
    SB  = get("SB")
    CS  = get("CS")
    QAB = get("QAB")
    HHB = get("HHB")
    LDc = get("LD")
    FBc = get("FB")
    GBc = get("GB")
    H_RISP  = get("H_RISP")
    AB_RISP = get("AB_RISP")
    PS  = get("PS")
    OUT2RBI = get("2OUTRBI")

    # If LD/FB/GB counts not present, try reconstruct from % and BIP totals
    BIP = AB - SO - HR + SF
    if LDc == 0 and "LD%" in tmp.columns:
        LDc = float(np.rint(_to_num(tmp["LD%"]).mean() * BIP)) if BIP > 0 else 0.0
    if FBc == 0 and "FB%" in tmp.columns:
        FBc = float(np.rint(_to_num(tmp["FB%"]).mean() * BIP)) if BIP > 0 else 0.0
    if GBc == 0 and "GB%" in tmp.columns:
        GBc = float(np.rint(_to_num(tmp["GB%"]).mean() * BIP)) if BIP > 0 else 0.0

    # Recompute team rates
    def sd(n,d):
        return 0.0 if d == 0 else round(float(n)/float(d), 3)

    AVG   = sd(H, AB)
    OBP   = sd(H + BB + HBP, AB + BB + HBP + SF)
    SLG   = sd(TB, AB)
    OPS   = round(OBP + SLG, 3)
    QABp  = sd(QAB, PA)
    HHBp  = sd(HHB, AB if AB > 0 else PA)
    LDp   = sd(LDc, (LDc + FBc + GBc))
    FBp   = sd(FBc, (LDc + FBc + GBc))
    GBp   = sd(GBc, (LDc + FBc + GBc))
    BABIP = sd(H - HR, AB - SO - HR + SF)
    BARISP= sd(H_RISP, AB_RISP)
    PS_PA = sd(PS, PA)
    BB_K  = 0.0 if SO == 0 else round(BB / SO, 3)
    Cp    = sd(AB - SO, AB)

    totals = {
        "Last": "TOTAL", "First": "",
        "PA": PA, "AB": AB, "AVG": AVG, "OBP": OBP, "OPS": OPS, "SLG": SLG, "H": H,
        "R": R, "RBI": RBI, "BB": BB, "SO": SO, "XBH": XBH, "2B": _2B, "3B": _3B, "HR": HR,
        "TB": TB, "SB": SB, "PS/PA": PS_PA, "BB/K": BB_K, "C%": Cp, "QAB": QAB, "QAB%": QABp,
        "HHB": HHB, "HHB%": HHBp, "LD%": LDp, "FB%": FBp, "GB%": GBp, "BABIP": BABIP,
        "BA/RISP": BARISP, "2OUTRBI": OUT2RBI
    }

    # Align totals dict to current df columns; fill missing keys with 0
    row = {c: totals.get(c, 0.0 if c not in ("Last","First") else ("TOTAL" if c=="Last" else "")) for c in tmp.columns}
    return pd.concat([tmp, pd.DataFrame([row])], ignore_index=True)

def _append_totals_row_pitching(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    for c in tmp.columns:
        if c not in ("Last","First"):
            tmp[c] = _to_num(tmp[c])

    # Sum counts
    get = lambda c: _to_num(tmp[c]).sum() if c in tmp.columns else 0.0
    ER = get("ER"); H = get("H"); R = get("R"); BB = get("BB"); SO = get("SO")
    HR = get("HR"); P = get("#P"); BF = get("BF"); HBP = get("HBP")
    CS = get("CS"); SB = get("SB"); BBS = get("BBS")
    Strikes = get("Strikes"); FPS = get("FirstPitchStrikes")
    FPSO = get("FPSO"); FPSH = get("FPSH")
    GB = get("GroundBalls"); FB = get("FlyBalls"); LD = get("LineDrives")
    HHB = get("HardHitBalls"); WEAK = get("WeakContact"); U3 = get("Under3Pitches")
    SM = get("SwingMisses")

    # Sum IP as decimal-thirds
    ip_dec = 0.0
    if "IP" in tmp.columns:
        ip_dec = float(_to_num(tmp["IP"]).apply(_convert_innings_value).sum())

    BIP = max(BF - SO - BB - HBP, 0.0)

    def sd(n,d):
        return 0.0 if d == 0 else round(float(n)/float(d), 3)

    ERA  = sd(ER * 9.0, ip_dec)
    WHIP = sd(BB + H, ip_dec)
    BB_INN = sd(BB, ip_dec)
    FIP = round((sd(13*HR + 3*(BB + HBP) - 2*SO, ip_dec) + 3.1), 3)

    S_pct   = sd(Strikes, P)
    FPS_pct = sd(FPS, BF)
    FPSO_pct= sd(FPSO, BF)
    FPSH_pct= sd(FPSH, BF)
    SM_pct  = sd(SM, P)
    LD_pct  = sd(LD, BIP)
    FB_pct  = sd(FB, BIP)
    GB_pct  = sd(GB, BIP)
    HHB_pct = sd(HHB, BIP)
    WEAK_pct= sd(WEAK, BIP)
    U3_pct  = sd(U3, BF)
    SB_pct  = 0.0 if (SB + CS) == 0 else round(SB / (SB + CS), 3)
    BAA     = sd(H, BF - BB - HBP)
    BABIP   = sd(H - HR, BF - SO - HR - BB - HBP)

    # Convert ip_dec to baseball-tenths style float
    ip_disp = _dec_to_baseball_tenths(ip_dec)

    totals = {
        "Last": "TOTAL", "First": "",
        "IP": ip_disp, "ERA": ERA, "WHIP": WHIP, "SO": SO, "K-L": get("K-L"), "H": H, "R": R, "ER": ER,
        "BB": BB, "BB/INN": BB_INN, "FIP": FIP, "S%": S_pct, "FPS%": FPS_pct, "FPSO%": FPSO_pct, "FPSH%": FPSH_pct,
        "BAA": BAA, "BBS": BBS, "SM%": SM_pct, "LD%": LD_pct, "FB%": FB_pct, "GB%": GB_pct, "BABIP": BABIP,
        "BA/RISP": 0.0, "CS": CS, "SB": SB, "SB%": SB_pct, "<3%": U3_pct, "HHB%": HHB_pct, "WEAK%": WEAK_pct,
        "#P": P, "BF": BF, "HBP": HBP
    }
    row = {c: totals.get(c, 0.0 if c not in ("Last","First") else ("TOTAL" if c=="Last" else "")) for c in tmp.columns}
    return pd.concat([tmp, pd.DataFrame([row])], ignore_index=True)

def _append_totals_row_fielding(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    for c in tmp.columns:
        if c not in ("Last","First"):
            tmp[c] = _to_num(tmp[c])
    get = lambda c: _to_num(tmp[c]).sum() if c in tmp.columns else 0.0
    TC = get("TC"); PO = get("PO"); A = get("A"); E = get("E"); DP = get("DP")
    FPCT = 0.0 if TC == 0 else round((PO + A) / TC, 3)
    totals = {"Last":"TOTAL","First":"", "TC":TC,"PO":PO,"A":A,"E":E,"DP":DP,"FPCT":FPCT}
    row = {c: totals.get(c, 0.0 if c not in ("Last","First") else ("TOTAL" if c=="Last" else "")) for c in tmp.columns}
    return pd.concat([tmp, pd.DataFrame([row])], ignore_index=True)

def _append_totals_row_catching(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    for c in tmp.columns:
        if c not in ("Last","First"):
            tmp[c] = _to_num(tmp[c])
    get = lambda c: _to_num(tmp[c]).sum() if c in tmp.columns else 0.0
    INN = get("INN"); PB = get("PB"); SB_ATT = get("SB-ATT"); CS = get("CS")
    CS_pct = 0.0 if SB_ATT == 0 else round(CS / SB_ATT, 3)
    totals = {"Last":"TOTAL","First":"", "INN":INN,"PB":PB,"SB-ATT":SB_ATT,"CS":CS,"CS%":CS_pct}
    row = {c: totals.get(c, 0.0 if c not in ("Last","First") else ("TOTAL" if c=="Last" else "")) for c in tmp.columns}
    return pd.concat([tmp, pd.DataFrame([row])], ignore_index=True)

def _append_totals(df: pd.DataFrame, tab_name: str) -> pd.DataFrame:
    if tab_name == "Hitting":
        return _append_totals_row_hitting(df)
    if tab_name == "Pitching":
        return _append_totals_row_pitching(df)
    if tab_name == "Fielding":
        return _append_totals_row_fielding(df)
    if tab_name == "Catching":
        return _append_totals_row_catching(df)
    return df

# =====================================================
# UI LAYOUT (same format/end product)
# =====================================================
st.set_page_config(page_title="EUCB Team Stats (PBDB-Logic Build)", layout="wide")

st.title("EUCB Team Stats (PBDB-Logic Build)")
st.caption("Cumulative or Series-aggregated view with percent-columns backfilled to counts before combining, then rates recalculated.")

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
            help="Series correspond to CSV base names (e.g., wake, jmu)."
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
