import json
import requests
import boto3
from datetime import datetime

# Configurations
BASE_URL = "https://e6uw49pbah.execute-api.us-east-1.amazonaws.com/dev"
TOKEN = "..." #Enter your token here
STATION_ID = "GDN_01"
BUCKET_NAME = "amzn-s3-gdansk-weather-bucket"

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

def collect_and_store_weather():
    # Requesting 10,000 records to gather full historical timeline
    endpoint = f"{BASE_URL}/weather/batch?station_id={STATION_ID}&limit=10000"
    
    print(f"Requesting data from API for station {STATION_ID}...")
    response = requests.get(endpoint, headers=headers)
    
    if response.status_code == 200:
        raw_data = response.json()
        records = raw_data.get("records", [])
        print(f"Success! API returned {len(records)} weather records.")
        
        # Unique filename using timestamp
        current_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_weather_{STATION_ID}_{current_time}.json"
        
        # 1. Save locally for verification
        local_path = f"data/bronze/{filename}"
        with open(local_path, "w") as f:
            json.dump(raw_data, f, indent=4)
        print(f"Saved copy locally to: {local_path}") # ! This line highlights that the limit is bounded to 500 values.
        
        # 2. Upload to S3 Bucket
        s3_key = f"data/bronze/{filename}"
        s3_client = boto3.client('s3')
        
        print(f"Uploading raw data to S3 bucket: {BUCKET_NAME}/{s3_key}...")
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(raw_data, indent=4),
            ContentType='application/json'
        )
        print("Upload complete! Raw data securely stored in S3.")
        
    elif response.status_code == 401:
        print("Unauthorized. Please verify your STUDENT_TOKEN_2026.")
    else:
        print(f"Failed to connect. Status Code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    collect_and_store_weather()