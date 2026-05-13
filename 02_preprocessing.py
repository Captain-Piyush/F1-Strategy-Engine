import fastf1
import pandas as pd
import os

# Configuration
CACHE_DIR = 'f1_cache'
fastf1.Cache.enable_cache(CACHE_DIR)

def preprocess_session(year, gp, session_type):
    """
    Standardizes F1 telemetry by aligning lap-level metadata with high-frequency 
    time-series data. Filters for competitive laps to ensure model quality.
    """
    print(f"\n>>> Extracting and aligning features for {year} {gp}...")
    
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    
    # Filter for 'QuickLaps' to remove pit entries, exits, and safety car periods
    # This ensures the model learns from representative racing pace
    laps = session.laps.pick_quicklaps()
    
    processed_data = []
    unique_drivers = laps['Driver'].unique()
    
    for driver in unique_drivers:
        # Updated from pick_driver to pick_drivers to satisfy latest API standards
        driver_laps = laps.pick_drivers(driver)
        
        for _, lap in driver_laps.iterrows():
            # Synchronize time-series telemetry with categorical lap data
            telemetry = lap.get_telemetry()
            
            telemetry['Driver'] = driver
            telemetry['LapNumber'] = lap['LapNumber']
            telemetry['Compound'] = lap['Compound']
            telemetry['TireLife'] = lap['TyreLife']
            telemetry['TrackStatus'] = lap['TrackStatus']
            
            processed_data.append(telemetry)

    # Consolidate into a single master feature matrix
    master_df = pd.concat(processed_data, ignore_index=True)
    
    # Feature pruning: Remove high-cardinality/non-predictive time markers
    # These columns consume VRAM without providing predictive value for the model
    drop_cols = ['Date', 'SessionTime', 'Source', 'Time']
    master_df = master_df.drop(columns=[c for c in drop_cols if c in master_df.columns])
    
    return master_df

if __name__ == "__main__":
    try:
        # Processing 2024 Saudi Arabia Race Data
        df = preprocess_session(2024, 'Saudi Arabia', 'R')
        
        # Local persistence for Feature Engineering stage
        output_path = os.path.join(CACHE_DIR, 'cleaned_telemetry.csv')
        df.to_csv(output_path, index=False)
        
        print(f"Success: Preprocessing Complete. Matrix Shape: {df.shape}")
        print(f"Data saved to {output_path}")
        
    except Exception as e:
        print(f"Error: Processing Pipeline Failed: {e}")