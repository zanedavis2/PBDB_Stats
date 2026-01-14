import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
st.set_page_config(page_title="PBDB Stats Aggregator", layout="wide", page_icon="⚾")

# Acronym Keys (Preserved for User Context)
HITTING_KEY = pd.DataFrame({
    "Acronym": ["PA", "AB", "H", "AVG", "OBP", "SLG", "OPS", "RBI", "R", "BB", "SO", "XBH", "2B", "3B", "HR", "TB", "SB", "PS/PA", "BB/K", "C%", "QAB", "QAB%", "HHB", "HHB%", "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "2OUTRBI"],
    "Meaning": ["Plate Appearances", "At-Bats", "Hits", "Batting Average", "On-Base Percentage", "Slugging Percentage", "On-base Plus Slugging", "Runs Batted In", "Runs Scored", "Walks", "Strikeouts", "Extra-Base Hits", "Doubles", "Triples", "Home Runs", "Total Bases", "Stolen Bases", "Pitches per PA", "Walk-to-Strikeout Ratio", "Contact %", "Quality At-Bats", "Quality At-Bat %", "Hard-Hit Balls", "Hard-Hit Ball %", "Line Drive %", "Fly Ball %", "Ground Ball %", "Batting Avg on Balls In Play", "Avg w/ RISP", "Two-Out RBIs"]
})

PITCHING_KEY = pd.DataFrame({
    "Acronym": ["IP", "ERA", "WHIP", "H", "R", "ER", "BB", "BB/INN", "SO", "K-L", "HR", "S%", "FPS%", "FPSO%", "FPSH%", "SM%", "<3%", "LD%", "FB%", "GB%", "HHB%", "WEAK%", "BBS", "BAA", "BABIP", "BA/RISP", "CS", "SB", "SB%", "FIP"],
    "Meaning": ["Innings Pitched", "Earned Run Average", "Walks+Hits per IP", "Hits Allowed", "Runs Allowed", "Earned Runs", "Walks", "Walks per Inning", "Strikeouts", "Strikeouts Looking", "Home Runs Allowed", "Strike %", "First-Pitch Strike %", "% FPS ABs ending in Out", "% FPS resulting in Hit", "Swinging Miss %", "% ABs ≤3 pitches", "Line Drive %", "Fly Ball %", "Ground Ball %", "Hard-Hit Ball %", "Weak Contact %", "Walks resulting in Run", "Batting Avg Against", "BABIP", "Avg w/ RISP", "Caught Stealing", "Stolen Bases Allowed", "Stolen Base %", "Fielding Independent Pitching"]
})

FIELDING_KEY = pd.DataFrame({
    "Acronym": ["TC", "A", "PO", "FPCT", "E", "DP"],
    "Meaning": ["Total Chances", "Assists", "Putouts", "Fielding Percentage", "Errors", "Double Plays"]
})

CATCHING_KEY = pd.DataFrame({
    "Acronym": ["INN", "PB", "SB-ATT", "CS", "CS%"],
    "Meaning": ["Innings Caught", "Passed Balls", "Stolen Base Attempts", "Caught Stealing", "Caught Stealing %"]
})

# ==========================================
# HELPER FUNCTIONS (Logic from pbdb_stats)
# ==========================================

def clean_df_structure(df):
    """
    Cleans the specific CSV format:
    1. Trims whitespace from headers and string columns.
    2. Identifies and removes the 'Totals' row and anything below it.
    3. Normalizes empty strings/NaNs.
    """
    # Clean Column Names
    df.columns = df.columns.astype(str).str.strip().str.replace(".1", "", regex=False)
    
    # Ensure Last/First exist
    if "Last" not in df.columns: df["Last"] = ""
    if "First" not in df.columns: df["First"] = ""

    # String cleanup
    df["Last"] = df["Last"].astype(str).str.strip().str.title()
    df["First"] = df["First"].astype(str).str.strip().str.title()
    
    # Replace artifacts with proper NaNs
    replace_vals = ["", "nan", "NaN", "None", "Nan"]
    df["Last"].replace(replace_vals, np.nan, inplace=True)
    
    # Find cutoff point (Totals row)
    # Logic: If 'Last' is NaN (after cleanup), it's likely the total row or empty space
    totals_idx = df.index[df["Last"].isna()]
    if len(totals_idx) > 0:
        first_total = totals_idx[0]
        df = df.iloc[:first_total].reset_index(drop=True)
        
    return df

def convert_innings_to_decimal(ip):
    """Converts x.1 (1/3) and x.2 (2/3) to proper floats for summation."""
    try:
        whole = int(ip)
        fraction = round((ip - whole) * 10)
        if fraction == 1:
            return whole + (1/3)
        elif fraction == 2:
            return whole + (2/3)
        else:
            return float(ip)
    except:
        return 0.0

def convert_decimal_to_innings_str(ip_val):
    """Converts decimal back to baseball string notation (e.g., 4.666 -> '4.2')."""
    whole = int(ip_val)
    remainder = ip_val - whole
    # account for floating point epsilon
    if remainder > 0.60:  # approx 2/3
        return f"{whole}.2"
    elif remainder > 0.30: # approx 1/3
        return f"{whole}.1"
    else:
        return f"{whole}.0"

def safe_numeric(series):
    """Coerces to numeric, filling NaNs with 0."""
    return pd.to_numeric(series, errors='coerce').fillna(0)

# ==========================================
# AGGREGATION LOGIC
# ==========================================

def process_pitching_files(uploaded_files):
    cols_to_keep = [
        "IP", "ER", "H", "BB", "R", "SO", "K-L", "HR", "#P", "BF", "HBP",
        "FPS%", "FPSO%", "FPSH%", "S%", "SM%", "LD%", "FB%", "GB%",
        "BABIP", "BA/RISP", "CS", "SB", "SB%", "<3%", "HHB%", "WEAK%", "BBS"
    ]
    
    dfs = []
    
    for uploaded_file in uploaded_files:
        try:
            # Read CSV (Header=1 is standard for these specific exports)
            df = pd.read_csv(uploaded_file, header=1)
            
            # Slice typical pitching columns if wide format, otherwise trust headers
            if df.shape[1] > 50:
                 # Logic from source: keep name cols (1,2) and pitching block (53-148)
                df = df.iloc[:, [1, 2] + list(range(53, min(148, df.shape[1])))]

            df = clean_df_structure(df)
            
            # Filter to relevant columns
            valid_cols = [c for c in cols_to_keep if c in df.columns]
            df = df[["Last", "First"] + valid_cols].copy()
            
            # Numeric conversion
            for col in valid_cols:
                df[col] = safe_numeric(df[col])

            # Convert Innings
            if "IP" in df.columns:
                df["IP"] = df["IP"].apply(convert_innings_to_decimal)

            # === REVERSE ENGINEER COUNTS FROM PERCENTAGES ===
            # This is the "Correct Logic" required. You cannot average percentages.
            # You must calculate the raw events, sum them, and recalculate.
            
            if "#P" in df.columns:
                df["Strikes"] = (df.get("S%", 0) * df["#P"] / 100).round(0)
                df["SwingMisses"] = (df.get("SM%", 0) * df["#P"] / 100).round(0)
            
            if "BF" in df.columns:
                df["FirstPitchStrikes"] = (df.get("FPS%", 0) * df["BF"] / 100).round(0)
                df["FPSO"] = (df.get("FPSO%", 0) * df["BF"] / 100).round(0)
                df["FPSH"] = (df.get("FPSH%", 0) * df["BF"] / 100).round(0)
                df["Under3Pitches"] = (df.get("<3%", 0) * df["BF"] / 100).round(0)

                # Batted Balls Denominator
                batted_balls = df["BF"] - df.get("SO", 0) - df.get("BB", 0) - df.get("HBP", 0)
                # Avoid negative batted balls from bad data
                batted_balls = batted_balls.clip(lower=0) 
                
                df["GroundBalls"] = (df.get("GB%", 0) * batted_balls / 100).round(0)
                df["FlyBalls"] = (df.get("FB%", 0) * batted_balls / 100).round(0)
                df["LineDrives"] = (df.get("LD%", 0) * batted_balls / 100).round(0)
                df["HardHitBalls"] = (df.get("HHB%", 0) * batted_balls / 100).round(0)
                df["WeakContact"] = (df.get("WEAK%", 0) * batted_balls / 100).round(0)

            dfs.append(df)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    if not dfs: return pd.DataFrame()

    # Aggregate
    combined = pd.concat(dfs, ignore_index=True)
    # Sum all numeric columns
    agg = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)

    # === RECALCULATE RATES ===
    # Avoid Divide by Zero
    ip = agg["IP"].replace(0, np.nan)
    bf = agg.get("BF", 0).replace(0, np.nan)
    num_p = agg.get("#P", 0).replace(0, np.nan)
    batted_balls = (agg.get("BF", 0) - agg.get("SO", 0) - agg.get("BB", 0) - agg.get("HBP", 0)).replace(0, np.nan)

    agg["ERA"] = (agg["ER"] * 9 / ip).round(2)
    agg["WHIP"] = ((agg["BB"] + agg["H"]) / ip).round(2)
    agg["BB/INN"] = (agg["BB"] / ip).round(2)
    
    # FIP Constant is typically 3.1 or 3.2 depending on league, using standard 3.1
    agg["FIP"] = (((13 * agg["HR"]) + (3 * agg["BB"]) - (2 * agg["SO"])) / ip + 3.1).round(2)

    # Re-calc Percentages
    agg["S%"] = (agg.get("Strikes", 0) / num_p * 100).round(1)
    agg["FPS%"] = (agg.get("FirstPitchStrikes", 0) / bf * 100).round(1)
    agg["FPSO%"] = (agg.get("FPSO", 0) / bf * 100).round(1)
    agg["FPSH%"] = (agg.get("FPSH", 0) / bf * 100).round(1)
    agg["SM%"] = (agg.get("SwingMisses", 0) / num_p * 100).round(1)
    agg["<3%"] = (agg.get("Under3Pitches", 0) / bf * 100).round(1)

    agg["GB%"] = (agg.get("GroundBalls", 0) / batted_balls * 100).round(1)
    agg["FB%"] = (agg.get("FlyBalls", 0) / batted_balls * 100).round(1)
    agg["LD%"] = (agg.get("LineDrives", 0) / batted_balls * 100).round(1)
    agg["HHB%"] = (agg.get("HardHitBalls", 0) / batted_balls * 100).round(1)
    agg["WEAK%"] = (agg.get("WeakContact", 0) / batted_balls * 100).round(1)

    agg["SB%"] = np.where((agg["SB"] + agg["CS"]) > 0, (agg["SB"] / (agg["SB"] + agg["CS"]) * 100), 0).round(1)
    
    # BAA
    at_bats_against = (agg.get("BF", 0) - agg.get("BB", 0) - agg.get("HBP", 0))
    agg["BAA"] = np.where(at_bats_against > 0, agg["H"] / at_bats_against, 0).round(3)

    # BABIP
    babip_denom = (agg.get("BF", 0) - agg.get("SO", 0) - agg.get("HR", 0) - agg.get("BB", 0) - agg.get("HBP", 0))
    agg["BABIP"] = np.where(babip_denom > 0, (agg["H"] - agg["HR"]) / babip_denom, 0).round(3)

    # Format IP back to string for display? Or keep decimal? 
    # Standard is usually decimal for math, string for display. 
    # We will keep decimal for sorting but user might prefer string. 
    # Let's add a Display IP column.
    agg["IP Display"] = agg["IP"].apply(convert_decimal_to_innings_str)

    # Final Column Ordering
    final_cols = ["Last", "First", "IP Display", "ERA", "WHIP", "H", "R", "ER", "BB", "SO", "K-L", "HR", "HBP", "FIP", "S%", "FPS%", "BAA", "BABIP", "SB", "CS", "SB%"]
    # Add others if they exist
    existing_extras = [c for c in ["LD%", "GB%", "FB%", "HHB%", "SM%"] if c in agg.columns]
    
    return agg[final_cols + existing_extras]

def process_hitting_files(uploaded_files):
    cols_to_keep = [
        "PA", "AB", "H", "BB", "HBP", "SF", "TB", "R", "RBI", "SO", "2B", "3B", "HR", "SB", "CS",
        "QAB", "HHB", "LD%", "FB%", "GB%", "H_RISP", "AB_RISP", "PS", "2OUTRBI", "XBH"
    ]
    
    dfs = []
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file, header=1)
            df = clean_df_structure(df)
            
            # Keep valid cols
            valid_cols = [c for c in cols_to_keep if c in df.columns]
            df = df[["Last", "First"] + valid_cols].copy()
            
            for c in valid_cols:
                df[c] = safe_numeric(df[c])
            
            # Reverse Engineer Hitting Counts
            # Some CSVs have raw counts for LD/FB/GB, some have %. 
            # If we only have %, we estimate count = % * AB (approx) or Batted Balls
            if "LD%" in df.columns and "AB" in df.columns:
                 # Standard approximation if raw BattedBall count isn't in source
                df["LD_Count"] = (df["LD%"] * df["AB"] / 100).round(0)
                df["GB_Count"] = (df["GB%"] * df["AB"] / 100).round(0)
                df["FB_Count"] = (df["FB%"] * df["AB"] / 100).round(0)
            
            dfs.append(df)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            
    if not dfs: return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    agg = combined.groupby(["Last", "First"], as_index=False).sum(numeric_only=True)

    # Derived Stats
    ab = agg["AB"].replace(0, np.nan)
    pa = agg["PA"].replace(0, np.nan)
    
    agg["AVG"] = (agg["H"] / ab).fillna(0).round(3)
    
    obp_denom = (agg["AB"] + agg["BB"] + agg.get("HBP", 0) + agg.get("SF", 0)).replace(0, np.nan)
    agg["OBP"] = ((agg["H"] + agg["BB"] + agg.get("HBP", 0)) / obp_denom).fillna(0).round(3)
    
    agg["SLG"] = (agg["TB"] / ab).fillna(0).round(3)
    agg["OPS"] = (agg["OBP"] + agg["SLG"]).round(3)
    
    agg["BB/K"] = (agg["BB"] / agg["SO"].replace(0, np.nan)).fillna(agg["BB"]).round(2)
    agg["C%"] = (1 - (agg["SO"] / ab)).fillna(0).round(3)
    
    if "QAB" in agg.columns:
        agg["QAB%"] = (agg["QAB"] / pa * 100).fillna(0).round(1)
        
    if "HHB" in agg.columns:
        agg["HHB%"] = (agg["HHB"] / ab * 100).fillna(0).round(1)

    # Recalculate Ball In Play Rates
    total_batted_est = agg.get("LD_Count", 0) + agg.get("GB_Count", 0) + agg.get("FB_Count", 0)
    total_batted_est = total_batted_est.replace(0, np.nan)
    
    if "LD_Count" in agg.columns:
        agg["LD%"] = (agg["LD_Count"] / total_batted_est * 100).fillna(0).round(1)
        agg["GB%"] = (agg["GB_Count"] / total_batted_est * 100).fillna(0).round(1)
        agg["FB%"] = (agg["FB_Count"] / total_batted_est * 100).fillna(0).round(1)
        
    if "AB_RISP" in agg.columns:
        agg["BA/RISP"] = (agg["H_RISP"] / agg["AB_RISP"].replace(0, np.nan)).fillna(0).round(3)

    # Order
    final_cols = ["Last", "First", "PA", "AB", "AVG", "OBP", "SLG", "OPS", "H", "R", "RBI", "HR", "2B", "3B", "BB", "SO", "HBP", "SB", "CS", "QAB%", "BB/K", "HHB%", "BABIP"]
    existing_cols = [c for c in final_cols if c in agg.columns]
    
    return agg[existing_cols]

def process_fielding_files(uploaded_files):
    cols_to_keep = ["TC", "A", "PO", "E", "DP"]
    dfs = []
    for f in uploaded_files:
        try:
            df = pd.read_csv(f, header=1)
            # Fielding usually at the end of the wide CSV
            if df.shape[1] > 140:
                df = df.iloc[:, [1, 2] + list(range(148, df.shape[1]))]
            
            df = clean_df_structure(df)
            valid = [c for c in cols_to_keep if c in df.columns]
            df = df[["Last", "First"] + valid].copy()
            for c in valid: df[c] = safe_numeric(df[c])
            dfs.append(df)
        except: continue
        
    if not dfs: return pd.DataFrame()
    
    agg = pd.concat(dfs).groupby(["Last", "First"], as_index=False).sum(numeric_only=True)
    
    if "TC" in agg.columns:
        agg["FPCT"] = ((agg["PO"] + agg["A"]) / agg["TC"].replace(0, np.nan)).fillna(0).round(3)
        
    return agg

def process_catching_files(uploaded_files):
    cols_to_keep = ["INN", "PB", "SB-ATT", "CS", "CS%"] # Note: CS% is derived, CS is raw
    # We need SB and CS raw counts to aggregate CS% correctly.
    # Usually catching files have "SB" and "CS" cols.
    
    dfs = []
    for f in uploaded_files:
        try:
            df = pd.read_csv(f, header=1)
            df = clean_df_structure(df)
            # Catching logic often simple sum
            valid = [c for c in ["INN", "PB", "CS", "SB"] if c in df.columns]
            df = df[["Last", "First"] + valid].copy()
            for c in valid: df[c] = safe_numeric(df[c])
            dfs.append(df)
        except: continue

    if not dfs: return pd.DataFrame()
    
    agg = pd.concat(dfs).groupby(["Last", "First"], as_index=False).sum(numeric_only=True)
    
    if "CS" in agg.columns and "SB" in agg.columns:
        agg["CS%"] = (agg["CS"] / (agg["CS"] + agg["SB"]).replace(0, np.nan) * 100).fillna(0).round(1)
        
    return agg


# ==========================================
# MAIN APP INTERFACE
# ==========================================

st.title("⚾ Advanced Baseball Stats Aggregator")
st.markdown("""
**Instructions:** Upload your CSV exports below. The app will automatically clean, merge, and recalculate all stats to ensure accuracy.
*Compatible with PBDB / GameChanger style exports.*
""")

uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Loaded {len(uploaded_files)} files.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Hitting", "Pitching", "Fielding", "Catching"])

    with tab1:
        st.subheader("Hitting Stats")
        df_hit = process_hitting_files(uploaded_files)
        if not df_hit.empty:
            st.dataframe(df_hit, use_container_width=True, hide_index=True)
            with st.expander("Hitting Glossary"):
                st.dataframe(HITTING_KEY, hide_index=True)
        else:
            st.info("No hitting data found in uploaded files.")

    with tab2:
        st.subheader("Pitching Stats")
        df_pitch = process_pitching_files(uploaded_files)
        if not df_pitch.empty:
            st.dataframe(df_pitch, use_container_width=True, hide_index=True)
            with st.expander("Pitching Glossary"):
                st.dataframe(PITCHING_KEY, hide_index=True)
        else:
            st.info("No pitching data found in uploaded files.")

    with tab3:
        st.subheader("Fielding Stats")
        df_field = process_fielding_files(uploaded_files)
        if not df_field.empty:
            st.dataframe(df_field, use_container_width=True, hide_index=True)
            with st.expander("Fielding Glossary"):
                st.dataframe(FIELDING_KEY, hide_index=True)
        else:
            st.info("No fielding data found.")

    with tab4:
        st.subheader("Catching Stats")
        df_catch = process_catching_files(uploaded_files)
        if not df_catch.empty:
            st.dataframe(df_catch, use_container_width=True, hide_index=True)
            with st.expander("Catching Glossary"):
                st.dataframe(CATCHING_KEY, hide_index=True)
        else:
            st.info("No catching data found.")

else:
    st.info("Please upload files to begin.")
