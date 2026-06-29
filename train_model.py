import io
import os
import pickle
import boto3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_save_and_evaluate():
    print("Starting Step 3: Model Training & Direct S3 Export...")
    
    bucket_name = "amzn-s3-gdansk-weather-bucket"
    silver_key = "data/silver/cleaned_weather_features.csv"
    
    # 1. Download the silver feature dataset from S3
    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=silver_key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        print(f"Successfully loaded {len(df)} records from S3 Silver Layer.")
    except Exception as e:
        print(f"Failed to read silver layer from S3: {e}")
        return

    # 2. Define Features and Target
    feature_cols = [
        'temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover', 'rain_mm',
        'temp_lag_1', 'temp_lag_2', 'temp_lag_3', 
        'temp_rolling_mean_1h', 'temp_rolling_std_1h',
        'hour_sin', 'hour_cos'
    ]
    
    X = df[feature_cols]
    y_1h = df['target_temp_1h']

    # 3. Chronological Train/Test Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_1h, y_test_1h = y_1h.iloc[:split_idx], y_1h.iloc[split_idx:]

    # 4. Train the Production Models
    print("Training Random Forest and Gradient Boosting architectures...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train_1h)
    
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbr.fit(X_train, y_train_1h)

    # 5. SERIALIZE & UPLOAD MODELS TO S3 GOLD LAYER
    print("Serializing and pushing trained model binaries directly to S3 Gold Layer...")
    models_to_upload = {
        "data/gold/random_forest_model.pkl": rf,
        "data/gold/gradient_boosting_model.pkl": gbr
    }
    
    for s3_key, model_obj in models_to_upload.items():
        try:
            # Pickling the model directly into an in-memory byte buffer
            model_buffer = io.BytesIO()
            pickle.dump(model_obj, model_buffer)
            model_buffer.seek(0)
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=model_buffer.getvalue()
            )
            print(f"Saved artifact: s3://{bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Failed to save {s3_key} to S3: {e}")

if __name__ == "__main__":
    train_save_and_evaluate()