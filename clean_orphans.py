#!/usr/bin/env python3
"""
HIV Orphans Data Cleaning Script

This script processes raw HIV orphans data from Excel,
standardizes columns, and saves a clean CSV file.
"""
import pandas as pd
import pathlib
import datetime
import os

# ------------------------------------
# Path Setup
# ------------------------------------
RAW_PATH = pathlib.Path("data_raw/HIV_Orphans_2024.xlsx")
CLEAN_PATH = pathlib.Path("data_clean/orphans_clean.csv")  # Changed to CSV instead of Parquet

# ------------------------------------
# A. Data Extraction
# ------------------------------------
def load_raw(filepath: pathlib.Path) -> pd.DataFrame:
    """Read the raw Excel file into a DataFrame, using the proper header row."""
    # Load the "Data" sheet with header row at index 1 (second row)
    df = pd.read_excel(filepath, sheet_name="Data", header=1)
    
    # Debug info
    print(f"   • Excel columns: {df.columns.tolist()[:5]}... (showing first 5)")
    print(f"   • First row sample: {df.iloc[0].tolist()[:3]}... (showing first 3 values)")
    
    return df

def capture_metadata(filepath: pathlib.Path):
    """Log file size, sheet names, and timestamp."""
    size = filepath.stat().st_size
    print(f"📄 File: {filepath}")
    print(f"   • Size: {size:,} bytes")
    
    # Get sheet names without full workbook load
    xl = pd.ExcelFile(filepath)
    print(f"   • Sheets: {xl.sheet_names}")
    print(f"   • Timestamp: {datetime.datetime.now()}\n")

# ------------------------------------
# B. Data Cleaning
# ------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to snake_case lower columns."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_$", "", regex=True)  # Remove trailing underscores
    )
    return df
def reshape_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide format (years as columns) to long format or handle already long data."""
    # Check if data is already in long format
    if "year" in df.columns and "value" in df.columns:
        print("   • Data is already in long format, renaming 'value' to 'orphans_0_17'")
        df_long = df.copy()
        df_long["orphans_0_17"] = df_long["value"]
        return df_long

    # Identify all year columns (strings of four digits)
    year_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit() and len(c) == 4]
    
    if not year_cols:
        print("⚠️ Warning: No year columns found. Checking column names:")
        print(f"   • Available columns: {df.columns.tolist()}")
        # Try to find year columns that might be integers or floats
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        # Convert numeric columns to strings for comparison
        potential_year_cols = []
        for c in numeric_cols:
            c_str = str(c)
            if c_str.isdigit() and len(c_str) == 4 and 1990 <= int(c_str) <= 2025:
                potential_year_cols.append(c_str)
        
        if potential_year_cols:
            print(f"   • Found potential year columns: {potential_year_cols}")
            year_cols = potential_year_cols
    
    # Get non-year columns as ID variables
    id_vars = [c for c in df.columns if c not in year_cols]
    
    if not year_cols:
        print("❌ Error: No year columns found to melt. Returning original dataframe.")
        if "year" not in df.columns:
            df["year"] = None  # Add empty year column
        if "orphans_0_17" not in df.columns:
            df["orphans_0_17"] = None  # Add empty orphans column
        return df
    
    print(f"   • Found {len(year_cols)} year columns from {min(year_cols)} to {max(year_cols)}")
    
    # Melt wide -> long
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year_str",  # temporary column
        value_name="orphans_0_17"
    )
    
    # Convert year to integer
    df_long["year"] = pd.to_numeric(df_long["year_str"], errors="coerce")
    df_long = df_long.drop(columns=["year_str"])
    
    print(f"   • Reshaped to {len(df_long):,} rows")
    
    return df_long

def filter_rows(df: pd.DataFrame, min_year: int = 2000, max_year: int = 2024) -> pd.DataFrame:
    """Keep only rows in [min_year, max_year]."""
    df = df.copy()
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
    
    # Drop "World" totals if a country column exists
    if "country" in df.columns:
        df = df[~df["country"].str.contains(r"world|total", case=False, na=False)]
    
    return df

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric columns, handling commas and converting to proper types."""
    df = df.copy()
    
    # Clean orphans count
    if "orphans_0_17" in df.columns:
        df["orphans_0_17"] = (
            df["orphans_0_17"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
    
    # Clean other numeric columns if they exist
    for col in ["hiv_prevalence", "population"]:
        if col in df.columns:
            if df[col].dtype == object:  # String column with possible commas
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                )
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing targets; handle other missing values."""
    df = df.copy()
    
    # Drop rows missing our target
    df = df.dropna(subset=["orphans_0_17"])
    
    # Optional: Impute other columns if needed
    # if "hiv_prevalence" in df.columns:
    #     df["hiv_prevalence"] = df["hiv_prevalence"].fillna(df["hiv_prevalence"].median())
    
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate country-year combinations."""
    # Use iso3 if it exists, otherwise use country
    id_col = "iso3" if "iso3" in df.columns else "country"
    
    # Make sure we have the column we need
    if id_col not in df.columns:
        print(f"⚠️ Warning: Cannot deduplicate without {id_col} column")
        return df
    
    # Check for duplicates
    dupes = df.duplicated(subset=[id_col, "year"], keep=False)
    if dupes.any():
        print(f"🔍 Found {dupes.sum()} duplicate {id_col}-year rows")
    
    # Drop duplicates
    df_unique = df.drop_duplicates(subset=[id_col, "year"])
    return df_unique

def perform_sanity_checks(df: pd.DataFrame):
    """Run validation checks on the cleaned data."""
    print("\n📊 Data summary:")
    print(f"   • Rows: {len(df):,}")
    print(f"   • Years: {df['year'].min()} to {df['year'].max()}")
    
    # Check for extreme values
    if "orphans_0_17" in df.columns:
        print(f"   • Orphans range: {df['orphans_0_17'].min():,.0f} to {df['orphans_0_17'].max():,.0f}")
    
    # Check for null values
    nulls = df.isnull().sum()
    if nulls.any():
        print("\n⚠️ Missing values found:")
        for col, count in nulls[nulls > 0].items():
            print(f"   • {col}: {count:,} missing values")
    else:
        print("\n✅ No missing values in final dataset")

# ------------------------------------
# C. Save Cleaned Data
# ------------------------------------
def save_clean(df: pd.DataFrame, outpath: pathlib.Path):
    """Write cleaned DataFrame to CSV."""
    # Ensure directory exists
    outpath.parent.mkdir(exist_ok=True, parents=True)
    
    # Check if dataframe is empty
    if len(df) == 0:
        print("\n⚠️ Warning: Attempting to save an empty dataframe!")
        print("   This suggests an issue in the data cleaning process.")
        
        # Save the file anyway (for debugging)
        df.to_csv(outpath, index=False)
        print(f"\n⚠️ Empty dataset saved to {outpath}")
        return
    
    # Save the file
    df.to_csv(outpath, index=False)
    print(f"\n✅ Cleaned data saved to {outpath} ({len(df):,} rows)")

# ------------------------------------
# Main Execution
# ------------------------------------
def main():
    """Execute the full data cleaning pipeline."""
    print("🔄 Starting HIV Orphans data cleaning process...\n")
    
    # A. Extraction
    capture_metadata(RAW_PATH)
    df_raw = load_raw(RAW_PATH)
    print(f"📥 Loaded raw data: {len(df_raw):,} rows, {len(df_raw.columns):,} columns")
    
    # B. Cleaning
    print("\n🧹 Cleaning data...")
    
    # After each step, check if we still have data
    df_norm = normalize_columns(df_raw)
    print(f"   • After normalization: {len(df_norm):,} rows")
    
    df_long = reshape_to_long(df_norm)
    print(f"   • After reshaping: {len(df_long):,} rows")
    
    # If we lost all data in reshaping, try to debug
    if len(df_long) == 0:
        print("\n❌ ERROR: Lost all data during reshaping!")
        print("Saving the normalized data for debugging...")
        debug_path = pathlib.Path("data_clean/debug_normalized.csv")
        df_norm.to_csv(debug_path, index=False)
        print(f"Debug data saved to {debug_path}")
        
        # Try to continue with original data as a fallback
        print("\n🔄 Attempting fallback approach...")
        # Skip the reshape and just use the normalized data
        df_filtered = df_norm
    else:
        df_filtered = filter_rows(df_long)
        print(f"   • After filtering: {len(df_filtered):,} rows")
    
    df_clean_nums = clean_numeric_columns(df_filtered)
    print(f"   • After numeric cleaning: {len(df_clean_nums):,} rows")
    
    df_no_missing = handle_missing(df_clean_nums)
    print(f"   • After handling missing values: {len(df_no_missing):,} rows")
    
    df_unique = remove_duplicates(df_no_missing)
    print(f"   • After removing duplicates: {len(df_unique):,} rows")
    
    # C. Validation and Save
    perform_sanity_checks(df_unique)
    save_clean(df_unique, CLEAN_PATH)
    
    print("\n✨ Data cleaning complete!")

if __name__ == "__main__":
    main()