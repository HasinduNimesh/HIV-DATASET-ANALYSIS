import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import os
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="HIV Orphans Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Function to download population data from World Bank API
def download_population_data():
    """Download population data from World Bank API for countries in our dataset"""
    try:
        # Create data_raw directory if it doesn't exist
        os.makedirs("data_raw", exist_ok=True)
        
        # Load country list from orphans data to know which countries we need
        orphans_df = pd.read_csv("data_clean/orphans_clean.csv")
        countries = orphans_df['iso3'].unique().tolist()
        
        # We'll need to get data for a range of years that matches our orphans data
        min_year = orphans_df['year'].min()
        max_year = orphans_df['year'].max()
        
        # Create dataframe to store population data
        pop_data = []
        
        with st.spinner(f"Downloading population data for {len(countries)} countries..."):
            progress_bar = st.progress(0)
            
            # World Bank API endpoint for population data
            base_url = "http://api.worldbank.org/v2/country/"
            indicator = "SP.POP.TOTL"  # Population, total
            
            # Process countries in batches to avoid API limits
            for i, country_code in enumerate(countries):
                # Construct API URL
                url = f"{base_url}{country_code}/indicator/{indicator}?format=json&date={min_year}:{max_year}"
                
                # Make API request
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # The API returns a list with metadata at index 0 and data at index 1
                    if len(data) > 1 and data[1]:
                        for entry in data[1]:
                            if entry['value'] is not None:
                                pop_data.append({
                                    'iso3': country_code,
                                    'year': int(entry['date']),
                                    'population': int(entry['value'])
                                })
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(countries))
                
                # Slight delay to avoid hitting API limits
                if i % 10 == 0:
                    time.sleep(0.5)
        
        # Convert to dataframe
        population_df = pd.DataFrame(pop_data)
        
        # Save to CSV
        output_path = "data_raw/population.csv"
        population_df.to_csv(output_path, index=False)
        
        return True, f"Successfully downloaded population data for {len(countries)} countries"
    
    except Exception as e:
        return False, f"Error downloading population data: {str(e)}"

# Load your model and results
try:
    model = joblib.load("results/rf_model.joblib")
    metrics = pd.read_csv("results/model_metrics.csv")
    
    # Load main orphans data
    data = pd.read_csv("data_clean/orphans_clean.csv")
    
    # Load GDP data and merge it properly
    gdp_data = pd.read_csv("data_raw/gdp_per_capita.csv")
    
    # Merge with main data - use outer merge to keep all data
    data = data.merge(gdp_data, on=['iso3', 'year'], how='left')
    
    # Check if we have population data file
    population_file = "data_raw/population.csv"
    if os.path.exists(population_file):
        # Load and merge population data
        pop_data = pd.read_csv(population_file)
        data = data.merge(pop_data, on=['iso3', 'year'], how='left')
        st.success("✅ Using actual World Bank population data")
    
    # If population is not in data, create mock population data
    if 'population' not in data.columns or data['population'].isna().all():
        # Create informative warning instead of just info
        st.warning("⚠️ Using estimated population data - for demonstration only")
        
        # Group by country to get reasonable population sizes
        country_factors = data.groupby('iso3')['orphans_0_17'].mean().reset_index()
        country_factors['factor'] = country_factors['orphans_0_17'] / country_factors['orphans_0_17'].min()
        factor_dict = dict(zip(country_factors['iso3'], country_factors['factor']))
        
        # Create population based on orphan count (larger countries have more orphans)
        data['population'] = data.apply(
            lambda row: int(5000000 * factor_dict.get(row['iso3'], 1)), axis=1
        )
    
    # Now filter to complete data
    required_columns = ['gdp_per_capita', 'population', 'orphans_0_17']
    complete_data = data.dropna(subset=required_columns).copy()
    valid_countries = sorted(complete_data["iso3"].unique())
    
    
    # Add title and description
    st.title("HIV Orphans Prediction Dashboard")
    st.markdown("""
    This dashboard forecasts the number of HIV orphans based on economic and population trends.
    Select a country and adjust parameters to generate predictions.
    """)
    
    # Show sidebar with model performance
    with st.sidebar:
        st.subheader("Model Performance")
        st.dataframe(metrics)
        
        st.subheader("Data Quality")
        st.info(f"Countries with complete data: {len(valid_countries)} of {len(data['iso3'].unique())}")
        
        if st.checkbox("Show countries with complete data"):
            st.write(valid_countries)
        
        # Population data section
        st.subheader("Population Data")
        if not os.path.exists(population_file):
            st.warning("Using estimated population data")
            if st.button("Download Real Population Data"):
                success, message = download_population_data()
                if success:
                    st.success(message)
                    st.info("Please reload the app to use the downloaded data")
                else:
                    st.error(message)
        else:
            st.success("Using real population data")
            last_modified = datetime.fromtimestamp(os.path.getmtime(population_file))
            st.info(f"Last updated: {last_modified.strftime('%Y-%m-%d')}")
            
            # Option to refresh data
            if st.button("Update Population Data"):
                success, message = download_population_data()
                if success:
                    st.success(message)
                    st.info("Please reload the app to use the updated data")
                else:
                    st.error(message)
        
        st.subheader("About")
        st.info("""
        This dashboard uses a Random Forest model trained on historical HIV orphans data.
        The model considers GDP per capita, population, and previous orphan counts to make predictions.
        """)
        
        with st.expander("About Population Data"):
            st.write("""
            Population data can come from two sources:
            1. **World Bank Data**: Real population figures downloaded from the World Bank API
            2. **Estimated Data**: When real data is unavailable, estimates are derived from orphan counts
            
            For more accurate forecasts, use the 'Download Real Population Data' button.
            """)
    
    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["Single Year Prediction", "Multi-Year Forecast"])
    
    with tab1:
        # Only show countries with complete data
        if not valid_countries:
            st.error("No countries with complete data found. Please check your dataset.")
        else:
            # Country selection
            selected_country = st.selectbox("Select a country", valid_countries, key="country_selector")
            country_data = complete_data[complete_data["iso3"] == selected_country]
            
            # Get the most recent data for the selected country to use as a basis for prediction
            if not country_data.empty:
                latest_data = country_data.sort_values('year', ascending=False).iloc[0]
                latest_year = int(latest_data['year'])
                
                st.subheader(f"Prediction for {selected_country}")
                
                # Create input fields for prediction
                col1, col2 = st.columns(2)
                
                with col1:
                    prediction_year = st.slider("Prediction Year:", 
                                            min_value=latest_year+1, 
                                            max_value=latest_year+10, 
                                            value=latest_year+1)
                    
                    gdp_growth = st.slider("Annual GDP Growth (%):", 
                                        min_value=-5.0, 
                                        max_value=10.0, 
                                        value=2.0,
                                        step=0.5)
                
                with col2:
                    population_growth = st.slider("Annual Population Growth (%):", 
                                             min_value=0.0, 
                                             max_value=5.0, 
                                             value=1.0,
                                             step=0.1)
                    
                    # Get latest orphan counts for lag features
                    orphans_recent = latest_data['orphans_0_17']
                    st.metric("Latest Known Orphans Count", f"{int(orphans_recent):,}")
                
                # When the user wants to make a prediction
                if st.button("Generate Prediction"):
                    try:
                        # Double-check all required data is present
                        for col in required_columns:
                            if pd.isna(latest_data[col]):
                                st.error(f"Missing {col} data for {selected_country}.")
                                st.stop()
                        
                        # Calculate future GDP and population based on growth rates
                        years_ahead = prediction_year - latest_year
                        gdp_future = latest_data['gdp_per_capita'] * ((1 + gdp_growth/100) ** years_ahead)
                        pop_future = latest_data['population'] * ((1 + population_growth/100) ** years_ahead)
                        
                        # Create a DataFrame with the necessary features for prediction
                        prediction_df = pd.DataFrame({
                            'gdp_per_capita': [gdp_future],
                            'population': [pop_future],
                            'orphans_0_17_lag1': [orphans_recent],
                            'orphans_0_17_lag2': [orphans_recent],
                            'orphans_0_17_lag3': [orphans_recent]
                        })
                        
                        # Make prediction
                        predicted_orphans = model.predict(prediction_df)[0]
                        
                        # Display prediction
                        st.success(f"Predicted orphans in {prediction_year}: **{int(predicted_orphans):,}**")
                        
                        # Show projected change
                        change = predicted_orphans - orphans_recent
                        change_pct = (change / orphans_recent) * 100
                        
                        if change >= 0:
                            st.metric("Projected Change", 
                                    f"+{int(change):,}", 
                                    f"+{change_pct:.1f}%")
                        else:
                            st.metric("Projected Change", 
                                    f"{int(change):,}", 
                                    f"{change_pct:.1f}%")
                        
                        # Visualization of historical + prediction
                        forecast_df = country_data.copy()
                        forecast_row = pd.DataFrame({
                            'iso3': [selected_country],
                            'year': [prediction_year],
                            'orphans_0_17': [predicted_orphans]
                        })
                        forecast_df = pd.concat([forecast_df, forecast_row])
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Plot historical data and prediction
                        sns.lineplot(data=forecast_df[forecast_df.year <= latest_year], 
                                    x='year', y='orphans_0_17', 
                                    color='blue', label='Historical')
                        
                        # Highlight the prediction
                        sns.lineplot(data=forecast_df[forecast_df.year >= latest_year], 
                                    x='year', y='orphans_0_17', 
                                    color='red', marker='o', linestyle='--', 
                                    label='Prediction')
                        
                        plt.title(f"Orphans Forecast for {selected_country}")
                        plt.ylabel("Number of orphans")
                        plt.grid(True)
                        plt.legend()
                        
                        st.pyplot(fig)
                        
                        # Fix the assumptions table - convert all values to strings
                        st.subheader("Forecast Assumptions")
                        assumptions = pd.DataFrame({
                            'Parameter': ['GDP per Capita Growth', 'Population Growth', 'Years Ahead', 'Model Type'],
                            'Value': [f'{gdp_growth}% annually', f'{population_growth}% annually', str(years_ahead), 'Random Forest']
                        })
                        st.table(assumptions)
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.info("There was a problem with the prediction. Please try a different country or contact support.")
                
            else:
                st.warning("No data available for selected country. Please choose another country.")
    
    with tab2:
        # Multi-year forecast
        st.subheader("Multi-Year Forecast")
        
        # Country selection for multi-year forecast - only show countries with complete data
        if valid_countries:
            selected_country_multi = st.selectbox("Select a country", valid_countries, key="multi_country_selector")
            country_data_multi = complete_data[complete_data["iso3"] == selected_country_multi]
            
            if not country_data_multi.empty:
                latest_data_multi = country_data_multi.sort_values('year', ascending=False).iloc[0]
                latest_year_multi = int(latest_data_multi['year'])
                
                # Parameters for multi-year forecast
                col1_multi, col2_multi = st.columns(2)
                
                with col1_multi:
                    forecast_years = st.slider("Forecast horizon (years):", 
                                             min_value=1, 
                                             max_value=10, 
                                             value=5)
                    
                    gdp_growth_multi = st.slider("Annual GDP Growth (%):", 
                                              min_value=-5.0, 
                                              max_value=10.0, 
                                              value=2.0,
                                              step=0.5,
                                              key="gdp_multi")
                
                with col2_multi:
                    population_growth_multi = st.slider("Annual Population Growth (%):", 
                                                     min_value=0.0, 
                                                     max_value=5.0, 
                                                     value=1.0,
                                                     step=0.1,
                                                     key="pop_multi")
                    
                    orphans_recent_multi = latest_data_multi['orphans_0_17']
                    st.metric("Latest Known Orphans Count", f"{int(orphans_recent_multi):,}")
                
                # Generate multi-year forecast
                if st.button("Generate Multi-Year Forecast"):
                    try:
                        # Double-check all required data is present
                        for col in required_columns:
                            if pd.isna(latest_data_multi[col]):
                                st.error(f"Missing {col} data for {selected_country_multi}.")
                                st.stop()
                        
                        # Create placeholder for forecast results
                        forecast_results = []
                        
                        # Start with the most recent known values
                        current_gdp = latest_data_multi['gdp_per_capita']
                        current_pop = latest_data_multi['population']
                        
                        # For lag features, start with the most recent known values
                        lag1 = orphans_recent_multi
                        lag2 = orphans_recent_multi
                        lag3 = orphans_recent_multi
                        
                        # Generate year-by-year predictions
                        for i in range(1, forecast_years + 1):
                            # Calculate GDP and population for this year
                            current_gdp *= (1 + gdp_growth_multi/100)
                            current_pop *= (1 + population_growth_multi/100)
                            
                            # Create prediction dataframe
                            pred_df = pd.DataFrame({
                                'gdp_per_capita': [current_gdp],
                                'population': [current_pop],
                                'orphans_0_17_lag1': [lag1],
                                'orphans_0_17_lag2': [lag2],
                                'orphans_0_17_lag3': [lag3]
                            })
                            
                            # Make prediction
                            predicted = model.predict(pred_df)[0]
                            
                            # Store result
                            forecast_results.append({
                                'iso3': selected_country_multi,
                                'year': latest_year_multi + i,
                                'orphans_0_17': predicted,
                                'gdp_per_capita': current_gdp,
                                'population': current_pop
                            })
                            
                            # Update lag features for next iteration
                            lag3 = lag2
                            lag2 = lag1
                            lag1 = predicted
                        
                        # Convert results to DataFrame
                        forecast_df_multi = pd.DataFrame(forecast_results)
                        
                        # Display results table
                        st.subheader("Forecast Results")
                        display_cols = ['year', 'orphans_0_17', 'gdp_per_capita', 'population']
                        st.dataframe(forecast_df_multi[display_cols].set_index('year'))
                        
                        # Visualize forecast
                        fig_multi, ax_multi = plt.subplots(figsize=(10, 6))
                        
                        # Plot historical data
                        sns.lineplot(data=country_data_multi, 
                                   x='year', y='orphans_0_17', 
                                   color='blue', label='Historical')
                        
                        # Plot forecast
                        sns.lineplot(data=forecast_df_multi, 
                                   x='year', y='orphans_0_17', 
                                   color='red', marker='o', linestyle='--', 
                                   label='Forecast')
                        
                        plt.title(f"Multi-Year Orphans Forecast for {selected_country_multi}")
                        plt.ylabel("Number of orphans")
                        plt.grid(True)
                        plt.legend()
                        
                        st.pyplot(fig_multi)
                        
                        # Calculate total change
                        total_change = forecast_df_multi.iloc[-1]['orphans_0_17'] - orphans_recent_multi
                        total_change_pct = (total_change / orphans_recent_multi) * 100
                        
                        st.subheader("Summary")
                        col1_summary, col2_summary = st.columns(2)
                        
                        with col1_summary:
                            st.metric(
                                f"Total Change ({latest_year_multi} to {latest_year_multi + forecast_years})", 
                                f"{int(total_change):+,}", 
                                f"{total_change_pct:.1f}%"
                            )
                        
                        with col2_summary:
                            # Calculate average annual change
                            avg_annual_change = total_change / forecast_years
                            avg_annual_pct = (avg_annual_change / orphans_recent_multi) * 100
                            st.metric(
                                "Average Annual Change", 
                                f"{int(avg_annual_change):+,}", 
                                f"{avg_annual_pct:.1f}%"
                            )
                        
                    except Exception as e:
                        st.error(f"Forecast error: {str(e)}")
                        st.info("There was a problem with the forecast. Please try a different country or contact support.")
            else:
                st.warning("No data available for selected country. Please choose another country.")
        else:
            st.error("No countries with complete data found. Please check your dataset.")
            
except Exception as e:
    st.error(f"Error loading required files: {str(e)}")
    st.info("""
    Make sure the following files exist:
    - results/rf_model.joblib
    - results/model_metrics.csv
    - data_clean/orphans_clean.csv
    
    You may need to run the predictive_modeling.py script first to generate these files.
    """)