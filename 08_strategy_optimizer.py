import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = 'f1_cache/degradation_metrics.csv'
TOTAL_LAPS = 50
PIT_LOSS = 22.0  # Approximate time lost driving through the Jeddah pit lane

def run_strategy_simulation(deg_df):
    """
    Simulates race times across varying pit stop laps to find the mathematically 
    optimal strategy window.
    """
    print(">>> Initializing Race Strategy Optimizer...")

    # Extract the average base pace and degradation slope for each compound
    summary = deg_df.groupby('Compound')[['Base_Pace', 'Degradation_Slope']].mean()

    def calculate_stint_time(compound, start_lap, end_lap):
        """Calculates total time taken to complete a specific stint."""
        laps = end_lap - start_lap
        if laps <= 0 or compound not in summary.index:
            return 0
        
        base = summary.loc[compound, 'Base_Pace']
        deg = summary.loc[compound, 'Degradation_Slope']
        
        # Arithmetic progression sum: total time = (laps * base) + (deg * sum of laps)
        # We use the sum of first N integers formula: n(n+1)/2
        total_time = (laps * base) + (deg * (laps * (laps + 1) / 2))
        return total_time

    strategies = []
    
    # Simulate pitting on every lap between Lap 5 and Lap 45
    print(">>> Simulating Medium -> Hard 1-Stop Strategies...")
    for pit_lap in range(5, 46):
        # Stint 1: Medium tires from Lap 0 to Pit Lap
        time_1 = calculate_stint_time('MEDIUM', 0, pit_lap)
        
        # Stint 2: Hard tires from Pit Lap to Checkered Flag
        time_2 = calculate_stint_time('HARD', pit_lap, TOTAL_LAPS)
        
        # Total Race Time = Stint 1 + Time Lost in Pits + Stint 2
        total_race_time = time_1 + PIT_LOSS + time_2
        
        strategies.append({
            'Pit_Lap': pit_lap, 
            'Total_Time_Seconds': total_race_time,
            'Total_Time_Formatted': pd.to_timedelta(total_race_time, unit='s')
        })

    # Convert to DataFrame for easy sorting and analysis
    strat_df = pd.DataFrame(strategies)
    
    # Find the strategy with the lowest total race time
    optimal = strat_df.loc[strat_df['Total_Time_Seconds'].idxmin()]

    print("\n--- Strategy Simulation Results ---")
    print(f"Optimal Pit Lap: Lap {int(optimal['Pit_Lap'])}")
    print(f"Predicted Total Race Time: {optimal['Total_Time_Formatted']}")
    
    print("\n--- Top 3 Pit Windows ---")
    top_3 = strat_df.sort_values('Total_Time_Seconds').head(3)
    for _, row in top_3.iterrows():
        print(f"Box Lap {int(row['Pit_Lap'])} | Total Time: {row['Total_Time_Formatted']}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run Script 07 first.")
    else:
        df = pd.read_csv(INPUT_FILE)
        run_strategy_simulation(df)