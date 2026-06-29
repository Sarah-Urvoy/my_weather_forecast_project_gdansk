import json
import os
import io
import boto3
import pandas as pd
import numpy as np

def process_raw_data_to_s3():
    print("Starting Step 2: Data Processing (Direct S3 Stream)...")
    
    # Configuration
    bucket_name = "amzn-s3-gdansk-weather-bucket"
    bronze_dir = "data/bronze/"
    silver_key = "data/silver/cleaned_weather_features.csv"
    
    # 1. Locate the latest raw JSON file locally to read
    files = sorted([f for f in os.listdir(bronze_dir) if f.endswith('.json')])
    if not files:
        print("No raw data files found locally in data/bronze/!")
        return
        
    latest_file = files[-1]
    print(f"Reading local raw file: {latest_file}")
    
    with open(os.path.join(bronze_dir, latest_file), 'r') as f:
        raw_data = json.load(f)
        
    # 2. Extract records into a DataFrame
    records = raw_data.get("records", [])
    df = pd.DataFrame(records)
    
    # 3. Data Cleaning
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.ffill()
    
    print(f"Loaded {len(df)} records. Applying feature engineering...")
    
    # 4. Feature Engineering
    df['target_temp_1h'] = df['temperature'].shift(-6)
    df['target_temp_6h'] = df['temperature'].shift(-36)
    
    df['temp_lag_1'] = df['temperature'].shift(1)
    df['temp_lag_2'] = df['temperature'].shift(2)
    df['temp_lag_3'] = df['temperature'].shift(3)
    
    df['temp_rolling_mean_1h'] = df['temperature'].rolling(window=6).mean()
    df['temp_rolling_std_1h'] = df['temperature'].rolling(window=6).std()
    
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    df = df.dropna().reset_index(drop=True)
    
    # 5. Direct Stream Upload to S3 (No local file created)
    print("Streaming finalized feature matrix directly to S3...")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3_client = boto3.client('s3')
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=silver_key,
            Body=csv_buffer.getvalue()
        )
        print(f"🚀 Success! Silver dataset uploaded straight to: s3://{bucket_name}/{silver_key}")
        print(f"Final dataset shape: {df.shape}")
    except Exception as e:
        print(f"❌ S3 Upload Failed! Error: {e}")

if __name__ == "__main__":
    process_raw_data_to_s3()