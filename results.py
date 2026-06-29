import io
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set styling for presentation-ready charts
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

def main():
    print("Downloading feature matrix from S3...")
    bucket_name = "amzn-s3-gdansk-weather-bucket"
    silver_key = "data/silver/cleaned_weather_features.csv"
    
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket_name, Key=silver_key)
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    
    # Define features and targets
    feature_cols = [
        'temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover', 'rain_mm',
        'temp_lag_1', 'temp_lag_2', 'temp_lag_3', 
        'temp_rolling_mean_1h', 'temp_rolling_std_1h',
        'hour_sin', 'hour_cos'
    ]
    
    X = df[feature_cols]
    y_1h = df['target_temp_1h']
    y_6h = df['target_temp_6h']
    
    # Chronological Train/Test Split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_1h, y_test_1h = y_1h.iloc[:split_idx], y_1h.iloc[split_idx:]
    y_train_6h, y_test_6h = y_6h.iloc[:split_idx], y_6h.iloc[split_idx:]
    
    # --- 1. TRAIN ALL MODELS ---
    print("Training models...")
    rf_1h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_1h)
    gbr_1h = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_1h)
    
    rf_6h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_6h)
    gbr_6h = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_6h)
    
    # --- 2. CALCULATE METRICS (1H vs 6H) ---
    # 1H Predictions
    p_ma_1h = X_test['temp_rolling_mean_1h']
    p_rf_1h = rf_1h.predict(X_test)
    p_gb_1h = gbr_1h.predict(X_test)
    
    # 6H Predictions (Using 6-hour rolling window baseline)
    p_ma_6h = df['temperature'].rolling(window=36).mean().iloc[split_idx:]
    p_rf_6h = rf_6h.predict(X_test)
    p_gb_6h = gbr_6h.predict(X_test)
    
    # Align index for 6H MA baseline due to rolling window drops
    valid_idx = p_ma_6h.dropna().index
    
    def get_scores(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))
    
    metrics = {
        "1H Horizon": {
            "Moving Average": get_scores(y_test_1h, p_ma_1h),
            "Random Forest": get_scores(y_test_1h, p_rf_1h),
            "Gradient Boosting": get_scores(y_test_1h, p_gb_1h)
        },
        "6H Horizon": {
            "Moving Average": get_scores(y_test_6h.loc[valid_idx], p_ma_6h.loc[valid_idx]),
            "Random Forest": get_scores(y_test_6h, p_rf_6h),
            "Gradient Boosting": get_scores(y_test_6h, p_gb_6h)
        }
    }
    
    print("\nTABLE 1: METRICS DATA FOR SLIDES")
    print("="*60)
    for horizon, mods in metrics.items():
        print(f"\n[{horizon}]")
        for m_name, (mae, rmse) in mods.items():
            print(f"  {m_name:<20} -> MAE: {mae:.2f}°C | RMSE: {rmse:.2f}°C")
            
    # --- 3. RECURSIVE MULTI-STEP SIMULATION (1 to 6 Hours Ahead) ---
    print("\nSimulating recursive forecast steps...")
    recursive_maes = []
    current_test_features = X_test.copy()
    
    for step in range(1, 7):
        # Predict 1 step ahead (10 mins * 6 steps = 1 hour incremental block)
        step_preds = gbr_1h.predict(current_test_features)
        
        # Calculate MAE against true future target at this step horizon
        true_target = df['temperature'].shift(-(step * 6)).iloc[split_idx:]
        valid_mask = true_target.notna()
        
        step_mae = mean_absolute_error(true_target[valid_mask], step_preds[valid_mask])
        recursive_maes.append(step_mae)
        
        # Workaround update: Inject prediction into temperature and adjust lag features
        current_test_features['temp_lag_3'] = current_test_features['temp_lag_2']
        current_test_features['temp_lag_2'] = current_test_features['temp_lag_1']
        current_test_features['temp_lag_1'] = current_test_features['temperature']
        current_test_features['temperature'] = step_preds

    # --- 4. GRAPH GENERATION ---
    print("\nPlotting presentation graphs...")
    
# Graph 1: Forecasting Comparison (Timeline Plot)
    plt.figure(figsize=(12, 5))
    subset_len = 80  # Plot slice for readability
    
    # y_test_1h is a Pandas Series, so .values is correct here
    plt.plot(y_test_1h.values[:subset_len], label="Actual Temperature", color='black', linewidth=2, linestyle='--')
    
    # p_ma_1h is a Pandas Series, so .values is correct here
    plt.plot(p_ma_1h.values[:subset_len], label="Moving Average Baseline", color='orange', alpha=0.7)
    
    plt.plot(p_rf_1h[:subset_len], label="Random Forest (1H)", color='blue', alpha=0.8)
    plt.plot(p_gb_1h[:subset_len], label="Gradient Boosting (1H)", color='green', alpha=0.9)
    plt.title("Forecast vs Actual Temperature Trajectory (1-Hour Horizon)")
    plt.xlabel("Timeline Samples (10-min intervals)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_1_forecast_comparison.png", dpi=300)
    plt.close()

    # Graph 2: Feature Importance (Input Parameter Influence)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    importances_config = [
        (rf_1h, "Random Forest (1H)", axes[0,0]),
        (gbr_1h, "Gradient Boosting (1H)", axes[0,1]),
        (rf_6h, "Random Forest (6H)", axes[1,0]),
        (gbr_6h, "Gradient Boosting (6H)", axes[1,1])
    ]
    for model, title, ax in importances_config:
        imp = model.feature_importances_
        idx = np.argsort(imp)[-7:]  # Show top 7 features
        ax.barh(np.array(feature_cols)[idx], imp[idx], color='teal', edgecolor='black', height=0.6)
        ax.set_title(f"Parameter Weights: {title}")
        ax.set_xlabel("Relative Influence Score")
    plt.tight_layout()
    plt.savefig("chart_2_parameter_influence.png", dpi=300)
    plt.close()

    # Graph 3: Recursive Multi-step Error Degradation Chart
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 7), recursive_maes, marker='o', color='red', linewidth=2, label="Recursive Error Accumulation")
    plt.title("Error Growth Across Multi-Hour Forecast Horizons")
    plt.xlabel("Forecasting Horizon (Hours Ahead)")
    plt.ylabel("Mean Absolute Error (MAE in °C)")
    plt.xticks(range(1, 7))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_3_recursive_error_growth.png", dpi=300)
    plt.close()
    
    print("Success!")

if __name__ == "__main__":
    main()