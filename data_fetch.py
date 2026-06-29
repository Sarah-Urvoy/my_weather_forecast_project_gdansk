import requests
import pandas as pd

# API Configuration Details
BASE_URL = "https://e6uw49pbah.execute-api.us-east-1.amazonaws.com/dev"
TOKEN = "..."  # Replace with your actual token
STATION_ID = "GDN_01"

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

def check_data_frequency():
    # Fetch a batch of records to inspect the time gap between them
    endpoint = f"{BASE_URL}/weather/batch?station_id={STATION_ID}&limit=100"
    
    print("Connecting to API...")
    response = requests.get(endpoint, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract the nested list of records using your exact JSON structure
        records = data.get("records", [])
        
        if not records:
            print("No records found in the response. Ensure the simulation has data.")
            return
            
        # Load the nested records into a Pandas DataFrame
        df = pd.DataFrame(records)
        
        # Convert timestamp strings to actual datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort chronologically to accurately measure consecutive differences
        df = df.sort_values('timestamp').reset_index(drop=True)
        # Calculate the time difference between consecutive rows
        df['time_diff'] = df['timestamp'].diff()
        
        # Find the most common time difference (the mode)
        most_common_frequency = df['time_diff'].mode()[0]
        
        print("\n--- API Connection Successful! ---")
        print(f"Sample response keys per record: {list(df.columns)}")
        print(f"Total records analyzed in this batch: {len(df)}")
        print(f"First timestamp in batch: {df['timestamp'].iloc[0]}")
        print(f"Last timestamp in batch: {df['timestamp'].iloc[-1]}")
        print(f"👉 DETECTED DATA FREQUENCY: Every {most_common_frequency}")
        print("-----------------------------------\n")
        
        return df
    elif response.status_code == 401:
        print("Error: Unauthorized. Check if your STUDENT_TOKEN_2026 is exact.")
    else:
        print(f"Failed to connect. Status Code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    df_sample = check_data_frequency()