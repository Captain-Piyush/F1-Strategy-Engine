import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = 'f1_cache/cleaned_telemetry.csv'
OUTPUT_FILE = 'f1_cache/final_features.csv'

def calculate_physics_features(df):
    """
    Derives physics-based features from raw telemetry.
    Note: FastF1 uses 'nGear' for gear information.
    """
    print(">>> Engineering physics features...")
    
    # Ensure data is sorted for consistent time-series calculations
    df = df.sort_values(by=['Driver', 'LapNumber', 'Distance'])
    
    # 1. Longitudinal Acceleration Proxy
    # Change in speed between samples
    df['Delta_Speed'] = df.groupby(['Driver', 'LapNumber'])['Speed'].diff().fillna(0)
    
    # 2. Braking Zones
    # FastF1 'Brake' is boolean; we use it with deceleration to find heavy braking
    df['Braking_Intensity'] = np.where((df['Brake'] == True) & (df['Delta_Speed'] < -5), 1, 0)
    
    # 3. Engine Stress
    # Fixed: Using 'nGear' instead of 'Gear'
    df['Engine_Stress'] = df['RPM'] / (df['nGear'] + 1)
    
    # 4. Pace Consistency
    # Rolling average of speed over the last 5 samples
    df['Rolling_Speed'] = df.groupby(['Driver', 'LapNumber'])['Speed'].transform(
        lambda x: x.rolling(window=5).mean()
    ).fillna(0)
    
    # 5. Tire Degradation Proxy
    # Cumulative distance traveled on current tire set
    df['Cumulative_Distance'] = df.groupby(['Driver', 'LapNumber', 'Compound'])['Distance'].transform('cumsum')
    
    return df

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run Script 02 first.")
    else:
        # Load the preprocessed matrix
        raw_df = pd.read_csv(INPUT_FILE)
        
        # Debugging: Print columns to verify availability
        print(f"Available Columns: {raw_df.columns.tolist()}")
        
        try:
            # Apply physics logic
            featured_df = calculate_physics_features(raw_df)
            
            # Save for Training
            featured_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Feature Engineering Complete. Features saved to {OUTPUT_FILE}")
            
        except KeyError as e:
            print(f"Column Mapping Error: {e}")
            print("Check if the column name exists in the 'Available Columns' list above.")