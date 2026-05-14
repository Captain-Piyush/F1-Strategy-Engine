import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os

# Configuration
INPUT_FILE = 'f1_cache/final_features.csv'
MODEL_PATH = 'f1_cache/strategy_model.json'

def train_model(df):
    """
    Trains an XGBoost Regressor on the GPU to predict car speed.
    """
    print(">>> Preparing data for GPU-accelerated training...")

    # 1. Categorical Encoding
    le = LabelEncoder()
    df['Compound_Enc'] = le.fit_transform(df['Compound'])
    df['Driver_Enc'] = le.fit_transform(df['Driver'])

    # 2. Feature Selection
    features = [
        'LapNumber', 'RPM', 'nGear', 'Throttle', 'Brake', 
        'Compound_Enc', 'TireLife', 'Delta_Speed', 
        'Braking_Intensity', 'Engine_Stress', 'Rolling_Speed',
        'Cumulative_Distance'
    ]
    
    X = df[features]
    y = df['Speed']

    # 3. Training/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f">>> Training XGBoost on GTX 1650 (CUDA Mode)...")
    
    # 4. Model Configuration
    # tree_method='hist' and device='cuda' offload the work to your 1650
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    # 5. Execution
    model.fit(X_train, y_train)

    # 6. Evaluation Metrics
    preds = model.predict(X_test)
    
    # FIXED: Modern Scikit-Learn 1.4+ uses root_mean_squared_error directly
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n--- Model Evaluation ---")
    print(f"RMSE (Average error in km/h): {rmse:.4f}")
    print(f"R2 Score (Confidence): {r2:.4f}")
    print(f"------------------------")

    return model

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
    else:
        data = pd.read_csv(INPUT_FILE)
        model = train_model(data)
        model.save_model(MODEL_PATH)
        print(f"Success: Model saved to {MODEL_PATH}")