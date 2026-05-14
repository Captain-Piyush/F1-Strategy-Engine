import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Configuration
MODEL_PATH = 'f1_cache/strategy_model.json'

def run_simulation():
    """
    Uses the trained model to predict performance in a hypothetical scenario.
    This is the core 'Engine' of the Strategy project.
    """
    # 1. Load the model
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    
    print(">>> F1 Strategy Engine: Inference Mode")
    
    # 2. Create a hypothetical scenario
    # Scenario: Lap 45 (Late race), Hard Tires, Full Throttle, various RPMs
    simulation_data = {
        'LapNumber': [45] * 5,
        'RPM': [10000, 11000, 12000, 13000, 14000],
        'nGear': [6, 6, 7, 7, 8],
        'Throttle': [100, 100, 100, 100, 100],
        'Brake': [0, 0, 0, 0, 0],
        'Compound_Enc': [0, 0, 0, 0, 0], # Assuming 0 is 'HARD' from earlier LabelEncoding
        'TireLife': [30, 30, 30, 30, 30],
        'Delta_Speed': [2, 2, 2, 2, 2],
        'Braking_Intensity': [0, 0, 0, 0, 0],
        'Engine_Stress': [0, 0, 0, 0, 0], # Placeholders - logic handles them
        'Rolling_Speed': [280, 290, 300, 310, 320],
        'Cumulative_Distance': [150000, 150000, 150000, 150000, 150000]
    }
    
    # Pre-calculate Engine_Stress for the simulation
    simulation_data['Engine_Stress'] = [r / (g + 1) for r, g in zip(simulation_data['RPM'], simulation_data['nGear'])]
    
    sim_df = pd.DataFrame(simulation_data)
    
    # 3. Predict
    # Note: Ensure feature order matches the training script exactly!
    features = [
        'LapNumber', 'RPM', 'nGear', 'Throttle', 'Brake', 
        'Compound_Enc', 'TireLife', 'Delta_Speed', 
        'Braking_Intensity', 'Engine_Stress', 'Rolling_Speed',
        'Cumulative_Distance'
    ]
    
    predictions = model.predict(sim_df[features])
    
    print("\n--- Simulation Results: Late Race Pace ---")
    for i, pred in enumerate(predictions):
        print(f"RPM: {simulation_data['RPM'][i]} | Gear: {simulation_data['nGear'][i]} | Predicted Speed: {pred:.2f} km/h")

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"Inference Failed: {e}")