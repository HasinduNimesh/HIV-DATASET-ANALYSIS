import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import random
import joblib
import yaml
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Set reproducible seeds
random.seed(42)
np.random.seed(42)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "data": {
        "orphans_file": "data_clean/orphans_clean.csv",
        "gdp_file": "data_raw/gdp_per_capita.csv",
        "output_dir": "figures/model"
    },
    "modeling": {
        "train_year_cutoff": 2020,
        "features": ["gdp_per_capita", "population", "orphans_lag1", "orphans_lag2", "orphans_lag3"],
        "target": "orphans_0_17",
        "random_state": 42,
        "n_jobs": 2  # Adjust based on your machine
    },
    "rf_params": {
        "n_estimators": [100, 300],
        "max_depth": [None, 5, 10]
    }
}

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file or use defaults."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            log.info(f"Loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        log.warning(f"Config file {config_path} not found. Using default configuration.")
        return DEFAULT_CONFIG

def setup_directories(config):
    """Create necessary directories."""
    dirs = [config["data"]["output_dir"], "results"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        log.info(f"Created directory: {d}")

def save_fig(fig, path, dpi=300):
    """Save figure with consistent settings."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    log.info(f"Saved figure to {path}")

def load_orphans_data(file_path):
    """Load the main orphans dataset."""
    log.info(f"Loading orphans data from {file_path}")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        log.error(f"Orphans data file not found at {file_path}")
        raise

def load_covariates(df, file_path):
    """Load and merge additional covariates."""
    log.info(f"Loading covariates from {file_path}")
    df_gdp = pd.read_csv(file_path)
    df_merged = df.merge(df_gdp, on=["iso3", "year"], how="left")
    log.info(f"Added GDP data: {df_merged['gdp_per_capita'].notna().sum()} rows with GDP values")
    return df_merged

def generate_mock_covariates(df):
    """Generate mock data for demonstration purposes."""
    log.warning("Creating mock GDP dataset for demonstration...")
    countries = df['iso3'].unique()
    years = range(int(df['year'].min()), int(df['year'].max()) + 1)
    mock_data = []
    
    for country in countries:
        base_gdp = np.random.randint(1000, 50000)
        for year in years:
            gdp = base_gdp * (1 + (year - 2000) * 0.02) * (1 + np.random.normal(0, 0.05))
            mock_data.append({'iso3': country, 'year': year, 'gdp_per_capita': gdp})
    
    df_gdp = pd.DataFrame(mock_data)
    df_merged = df.merge(df_gdp, on=["iso3", "year"], how="left")
    log.info("Added mock GDP data for demonstration purposes")
    return df_merged

def create_lag_features(df, target_col, lags):
    """Create lag features for time series data."""
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df.groupby("iso3")[target_col].shift(lag)
    log.info(f"Created lag features: {', '.join([f'{target_col}_lag{l}' for l in lags])}")
    return df

def create_population_features(df):
    """Create population-based features if population exists or mock data if not."""
    if "population" in df.columns:
        df["orphans_rate"] = df["orphans_0_17"] / df["population"] * 1000
        log.info("Calculated orphan rate per 1000 children")
    else:
        log.warning("Population data not found. Creating mock population data...")
        np.random.seed(42)
        for country in df['iso3'].unique():
            base_pop = np.random.randint(1_000_000, 100_000_000)
            country_mask = df['iso3'] == country
            for year in sorted(df[country_mask]['year'].unique()):
                year_growth = 1 + (np.random.random() * 0.03)  # 0-3% annual growth
                year_mask = df['year'] == year
                base_pop = base_pop * year_growth
                df.loc[country_mask & year_mask, 'population'] = base_pop
        
        df["orphans_rate"] = df["orphans_0_17"] / df["population"] * 1000
        log.info("Created mock population data and calculated orphan rate")
    
    return df

def impute_missing_values(df, numeric_cols):
    """Impute missing values using forward/backward fill within countries."""
    rows_before = len(df)
    
    # Try to recover GDP data with forward/backward fill within countries
    for col in numeric_cols:
        if col in df.columns:
            # Use transform instead of apply to maintain index alignment
            df[col] = df.groupby('iso3')[col].transform(
                lambda s: s.ffill().bfill()
            )
    
    log.info(f"Imputed missing values in {', '.join(numeric_cols)}")
    
    # For remaining missing values in critical columns, drop rows
    # Fix lag column names to match what was created
    features_needed = ["gdp_per_capita"] + [f"orphans_0_17_lag{l}" for l in (1,2,3)]
    df_clean = df.dropna(subset=features_needed)
    
    log.info(f"Dropped rows with missing values after imputation: {rows_before - len(df_clean)} rows removed")
    log.info(f"Final dataset: {len(df_clean)} rows with complete data")
    
    return df_clean

def prepare_data(config):
    """Master function for data preparation."""
    log.info("1. Loading and preparing data...")
    
    # Load main orphans data
    df = load_orphans_data(config["data"]["orphans_file"])
    
    # Add covariates (real or mock)
    try:
        df = load_covariates(df, config["data"]["gdp_file"])
    except FileNotFoundError:
        df = generate_mock_covariates(df)
    
    # Feature engineering
    df = create_lag_features(df, config["modeling"]["target"], lags=[1, 2, 3])
    df = create_population_features(df)
    
    # Impute missing values
    df = impute_missing_values(df, ["gdp_per_capita", "population"])
    
    return df

def train_test_split_by_year(df, cutoff_year, features, target):
    """Split data into train and test sets based on year."""
    train = df[df.year <= cutoff_year]
    test = df[df.year > cutoff_year]
    
    log.info(f"Train set: {len(train)} rows ({min(train.year) if not train.empty else 'N/A'} to {cutoff_year})")
    log.info(f"Test set: {len(test)} rows ({cutoff_year+1} to {max(test.year) if not test.empty else 'N/A'})")
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    return X_train, y_train, X_test, y_test, train, test

def evaluate_baseline(test_df, target, lag_col):
    """Evaluate baseline persistence model."""
    log.info("Evaluating baseline model...")
    y_pred_baseline = test_df[lag_col].values
    y_true = test_df[target].values
    
    mae_baseline = mean_absolute_error(y_true, y_pred_baseline)
    r2_baseline = r2_score(y_true, y_pred_baseline)
    
    log.info(f"Baseline MAE: {mae_baseline:.0f}, R²: {r2_baseline:.3f}")
    
    return y_pred_baseline, mae_baseline, r2_baseline

def build_model_pipeline(random_state):
    """Create a scikit-learn pipeline with preprocessing and model."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=random_state))
    ])

def train_model(X_train, y_train, param_grid, n_jobs, random_state):
    """Train model with hyperparameter tuning."""
    log.info("Training RandomForest model (this might take a few minutes)...")
    
    pipe = build_model_pipeline(random_state)
    
    # Adjust param_grid for pipeline
    pipeline_param_grid = {f'rf__{k}': v for k, v in param_grid.items()}
    
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(
        pipe, 
        pipeline_param_grid, 
        cv=tscv, 
        scoring="neg_mean_absolute_error", 
        n_jobs=n_jobs
    )
    
    grid.fit(X_train, y_train)
    
    log.info(f"Best parameters: {grid.best_params_}")
    
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    log.info(f"Model MAE: {mae:.0f}, R²: {r2:.3f}")
    
    return y_pred, mae, r2

def calculate_feature_importance(model, X_test, y_test, features, n_repeats=10, random_state=42):
    """Calculate and return permutation feature importance."""
    log.info("Calculating feature importance...")
    
    # Extract the RF from the pipeline
    rf = model.named_steps['rf']
    
    imp = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state)
    imp_df = pd.DataFrame({
        "feature": features,
        "importance": imp.importances_mean,
        "std": imp.importances_std
    }).sort_values("importance", ascending=False)
    
    log.info("Feature importance calculated")
    
    return imp_df, rf

def plot_predicted_vs_actual(y_test, y_pred, mae, output_dir):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
    plt.xlabel("Actual Orphans")
    plt.ylabel("Predicted Orphans")
    plt.title(f"RF Predictions (MAE={mae:.0f})")
    
    save_fig(plt.gcf(), f"{output_dir}/pred_vs_actual_rf.png")
    plt.close()

def plot_residuals(test_df, y_test, y_pred, output_dir):
    """Plot residuals over time and residual distribution."""
    resid = y_test - y_pred
    
    # Residuals over time
    plt.figure(figsize=(8, 4))
    plt.plot(test_df.year, resid, marker="o", linestyle="-", alpha=0.7)
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("Year")
    plt.ylabel("Residual (Actual–Pred)")
    plt.title("Time-Series Residuals (RF)")
    
    save_fig(plt.gcf(), f"{output_dir}/residuals_rf.png")
    plt.close()
    
    # Residual distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(resid, kde=True)
    plt.axvline(0, color="r", linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    
    save_fig(plt.gcf(), f"{output_dir}/residual_dist_rf.png")
    plt.close()
    
    # QQ plot for normality check
    from scipy import stats
    plt.figure(figsize=(6, 6))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    
    save_fig(plt.gcf(), f"{output_dir}/residual_qq_rf.png")
    plt.close()

def plot_feature_importance(imp_df, output_dir):
    """Plot feature importance with error bars."""
    plt.figure(figsize=(8, 5))
    
    # Extract feature names and importance values
    features = imp_df['feature'].tolist()
    importances = imp_df['importance'].tolist()
    
    # Check if 'std' is a column in the DataFrame
    if 'std' in imp_df.columns:
        stds = imp_df['std'].tolist()
        # Plot with error bars
        plt.barh(features, importances, xerr=stds, 
                 color="skyblue", ecolor="gray", capsize=5)
    else:
        # No error bars
        plt.barh(features, importances, color="skyblue")
    
    plt.xlabel("Mean Decrease in MAE")
    plt.title("Permutation Feature Importance with Standard Error")
    plt.tight_layout()
    
    save_fig(plt.gcf(), f"{output_dir}/feature_importance_rf.png")
    plt.close()
    
def plot_partial_dependence(model, X_test, top_features, feature_names, output_dir):
    """Plot partial dependence for top features."""
    log.info("Calculating partial dependence plots...")
    
    fig, axes = plt.subplots(1, len(top_features), figsize=(4*len(top_features), 4))
    if len(top_features) == 1:
        axes = [axes]
        
    # Create PDP display
    pdp = PartialDependenceDisplay.from_estimator(
        model, X_test, top_features, feature_names=feature_names,
        ax=axes, kind="both"
    )
    
    # Add median markers to show typical values
    for i, feature in enumerate(top_features):
        ax = axes[i]
        median_value = X_test[feature].median()
        ylim = ax.get_ylim()
        ax.axvline(median_value, color='r', linestyle='--', alpha=0.7)
        ax.text(median_value, ylim[0] + 0.9*(ylim[1]-ylim[0]), 
                f"Median\n{median_value:.1f}", 
                color='r', ha='center')
    
    plt.tight_layout()
    save_fig(fig, f"{output_dir}/pdp_top{len(top_features)}.png")
    plt.close()

def main():
    """Main execution flow."""
    # Load configuration
    config = load_config()
    setup_directories(config)
    
    # Set consistent plot style
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 10
    
    # Data preparation
    df = prepare_data(config)
    
    # Modeling preparation
    log.info("\n2. Setting up modeling strategy...")
    log.info("• Baseline: persistence model (previous year's value)")
    log.info("• RandomForest with scikit-learn Pipeline and time series CV")
    
    # Update features to match actual column names in DataFrame
    # This fixes the mismatch between config and actual column names
    target = config["modeling"]["target"]
    features = []
    for feature in config["modeling"]["features"]:
        if feature.startswith("orphans_lag"):
            lag_num = feature.split("lag")[1]
            features.append(f"{target}_lag{lag_num}")
        else:
            features.append(feature)
    
    log.info(f"Using features: {features}")
    
    # Train/test split
    log.info("\n3. Training and validating models...")
    X_train, y_train, X_test, y_test, train, test = train_test_split_by_year(
        df, config["modeling"]["train_year_cutoff"], features, target
    )
    
    # Baseline evaluation
    y_pred_baseline, mae_baseline, r2_baseline = evaluate_baseline(
        test, target, f"{target}_lag1"
    )
    
    # Model training and evaluation
    model = train_model(
        X_train, y_train, 
        config["rf_params"], 
        config["modeling"]["n_jobs"],
        config["modeling"]["random_state"]
    )
    
    y_pred_rf, mae_rf, r2_rf = evaluate_model(model, X_test, y_test)
    
    # Save model for later use
    joblib.dump(model, "results/rf_model.joblib")
    log.info("Saved trained model to results/rf_model.joblib")
    
    # Results summary
    metrics = pd.DataFrame({
        "model": ["Baseline", "RandomForest"],
        "MAE": [mae_baseline, mae_rf],
        "R2": [r2_baseline, r2_rf]
    })
    
    log.info("\nModel Comparison Results:")
    log.info("\n" + metrics.to_string(index=False))
    
    # Save metrics to CSV
    metrics.to_csv("results/model_metrics.csv", index=False)
    log.info("Saved metrics to results/model_metrics.csv")
    
    # Performance analysis
    log.info("\n4. Analyzing model performance...")
    
    # Create visualizations
    plot_predicted_vs_actual(y_test, y_pred_rf, mae_rf, config["data"]["output_dir"])
    plot_residuals(test, y_test, y_pred_rf, config["data"]["output_dir"])
    
    # Feature importance and interpretation
    log.info("\n5. Interpreting model...")
    imp_df, rf_model = calculate_feature_importance(
        model, X_test, y_test, features,
        n_repeats=10, 
        random_state=config["modeling"]["random_state"]
    )
    
    # Display feature importance
    log.info("\nFeature Importance:")
    log.info("\n" + imp_df.to_string(index=False))
    
    # Plot feature importance
    plot_feature_importance(imp_df, config["data"]["output_dir"])
    
    # Plot partial dependence for top 2 features
    top_features = imp_df.feature.iloc[:min(2, len(imp_df))].tolist()
    plot_partial_dependence(
        model, X_test, top_features, features, 
        config["data"]["output_dir"]
    )
    
    log.info("\n✅ Modeling complete! Check the 'figures/model' directory for outputs.")

if __name__ == "__main__":
    main()