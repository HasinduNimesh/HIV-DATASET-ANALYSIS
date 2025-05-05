import joblib
import pandas as pd

# Load your trained model
model = joblib.load("results/rf_model.joblib")

# Example: Create prediction data for 2024-2025 for all countries
# (You'd need to prepare this data with the same features as your training data)
future_df = pd.read_csv("data_clean/orphans_clean.csv")
future_countries = future_df['iso3'].unique()

# Create sample future data
prediction_data = []
for country in future_countries:
    # Get last known values from 2023
    last_known = future_df[(future_df.iso3 == country) & (future_df.year == 2023)].iloc[0]
    
    # For each future year
    for year in [2024, 2025]:
        # Create a row with the last known values as lag features
        prediction_data.append({
            'iso3': country,
            'year': year,
            'gdp_per_capita': last_known['gdp_per_capita'] * 1.02,  # Assume 2% growth
            'population': last_known['population'] * 1.01,  # Assume 1% growth
            'orphans_0_17_lag1': last_known['orphans_0_17'],
            'orphans_0_17_lag2': None,  # You'd need actual values
            'orphans_0_17_lag3': None   # You'd need actual values
        })

# Make predictions
# (You'd need to handle missing lags appropriately)