# scripts/clean_gdp.py

import pandas as pd
from pathlib import Path

# 1. Paths
INFILE = Path("data_raw/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_19346.csv")
OUTFILE = Path("data_raw/gdp_per_capita.csv")

# 2. Load (skip the 4-line metadata header)
df = pd.read_csv(INFILE, skiprows=4)

# 3. Normalize columns
df.columns = df.columns.str.strip()

# 4. Identify year columns (strings of 4 digits)
year_cols = [c for c in df.columns if c.isdigit() and len(c) == 4]
# We only need 2000–2023
year_cols = [c for c in year_cols if 2000 <= int(c) <= 2023]

# 5. Keep only Country Code + our year columns
df_small = df[["Country Code"] + year_cols].rename(columns={"Country Code":"iso3"})

# 6. Melt into long format
df_long = df_small.melt(
    id_vars="iso3",
    value_vars=year_cols,
    var_name="year",
    value_name="gdp_per_capita"
)

# 7. Clean up types
df_long["year"] = df_long["year"].astype(int)
df_long["gdp_per_capita"] = pd.to_numeric(df_long["gdp_per_capita"], errors="coerce")

# 8. Drop any rows where GDP is missing
df_long = df_long.dropna(subset=["gdp_per_capita"])

# 9. Save tidy CSV
df_long.to_csv(OUTFILE, index=False)
print(f"✅ Saved cleaned GDP data to {OUTFILE} ({len(df_long):,} rows)")
