import pandas as pd

# Use CSV file path instead of Parquet
CLEAN_PATH = "data_clean/orphans_clean.csv"  

# 1. Load only the first few rows to inspect columns and sample values
df_head = pd.read_csv(CLEAN_PATH).head(5)
print("=== Sample Rows ===")
print(df_head)

# 2. Check overall shape
df_shape = pd.read_csv(CLEAN_PATH).shape
print("\nDataset shape: rows =", df_shape[0], ", columns =", df_shape[1])

# 3. Year range
years = pd.read_csv(CLEAN_PATH, usecols=["year"])["year"]
print("\nYear range:", years.min(), "to", years.max())

# 4. Unique countries
iso3 = pd.read_csv(CLEAN_PATH, usecols=["iso3"])["iso3"]
print("Unique ISO3 count:", iso3.nunique())

# 5. Missing values per column
print("\nMissing values per column:")
print(pd.read_csv(CLEAN_PATH).isna().sum())