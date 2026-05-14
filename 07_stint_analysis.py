import fastf1
import pandas as pd
import numpy as np
import os

# Configuration
CACHE_DIR = 'f1_cache'
fastf1.Cache.enable_cache(CACHE_DIR)
OUTPUT_FILE = 'f1_cache/degradation_metrics.csv'

def calculate_degradation(year, gp, session_type):
    """
    Extracts stint lengths and calculates linear tire degradation 
    (seconds lost per lap) for each tire compound.
    """
    print(f"\n>>> Initializing Phase 2: Stint Analysis for {year} {gp}...")
    
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    
    laps = session.laps.pick_quicklaps()
    
    # Filter out anomalous laps (safety cars, VSC)
    # TrackStatus '1' indicates a clear, green-flag track
    clean_laps = laps[laps['TrackStatus'] == '1'].copy()
    
    # Convert LapTime timedelta to numerical seconds for mathematical modeling
    clean_laps['LapTime_Sec'] = clean_laps['LapTime'].dt.total_seconds()
    
    degradation_data = []
    unique_compounds = clean_laps['Compound'].dropna().unique()
    
    for compound in unique_compounds:
        compound_laps = clean_laps[clean_laps['Compound'] == compound]
        
        # We need enough data points to calculate a slope
        if len(compound_laps) < 10:
            continue
            
        print(f"Calculating degradation curve for {compound} compound...")
        
        # Group by driver and stint to find individual slopes
        for driver in compound_laps['Driver'].unique():
            driver_laps = compound_laps[compound_laps['Driver'] == driver]
            
            for stint in driver_laps['Stint'].unique():
                stint_laps = driver_laps[driver_laps['Stint'] == stint]
                
                # Minimum 5 laps in a stint to establish a valid degradation trend
                if len(stint_laps) >= 5:
                    # Calculate linear regression (slope) of LapTime over TireLife
                    x = stint_laps['TyreLife'].values
                    y = stint_laps['LapTime_Sec'].values
                    
                    # np.polyfit returns [slope, intercept]
                    try:
                        slope, intercept = np.polyfit(x, y, 1)
                    except:
                        continue
                        
                    degradation_data.append({
                        'Driver': driver,
                        'Compound': compound,
                        'Stint': stint,
                        'Stint_Length': len(stint_laps),
                        'Base_Pace': intercept,
                        'Degradation_Slope': slope
                    })

    # Compile the final matrix
    deg_df = pd.DataFrame(degradation_data)
    
    # Filter out outliers (slopes that are physically impossible in standard racing conditions)
    deg_df = deg_df[(deg_df['Degradation_Slope'] > -0.5) & (deg_df['Degradation_Slope'] < 2.0)]
    
    return deg_df

if __name__ == "__main__":
    try:
        # Processing 2024 Saudi Arabia Race Data
        deg_matrix = calculate_degradation(2024, 'Saudi Arabia', 'R')
        
        # Display aggregate analytics
        print("\n--- Average Degradation per Compound (Seconds/Lap) ---")
        summary = deg_matrix.groupby('Compound')['Degradation_Slope'].mean().reset_index()
        for _, row in summary.iterrows():
            print(f"{row['Compound']}: +{row['Degradation_Slope']:.3f} seconds per lap")
            
        print("------------------------------------------------------")
        
        deg_matrix.to_csv(OUTPUT_FILE, index=False)
        print(f"Success: Degradation matrix saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error: Pipeline Failed: {e}")