"""
APEX PREDATOR — apex_train.py
================================
Feature engineering + LightGBM model training.

Input:  f1_data_historical/historical_database.csv
Output: f1_data_historical/apex_model.pkl
        f1_data_historical/feature_importance.csv
        f1_data_historical/model_metadata.json

Run after apex_historical.py:
  python apex_train.py

Also run after each race to incorporate new 2026 data:
  python apex_train.py  (automatically finds and adds current season data)

The model predicts finishing position for each driver.
We then convert predicted positions to win probabilities via softmax.
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings("ignore")

HIST_PATH   = "f1_data_historical/historical_database.csv"
MODEL_PATH  = "f1_data_historical/apex_model.pkl"
META_PATH   = "f1_data_historical/model_metadata.json"
FEAT_PATH   = "f1_data_historical/feature_importance.csv"

CURRENT_YEAR = datetime.now().year
CURR_DB_PATH = f"f1_data_{CURRENT_YEAR}/database.csv"


# ── regulation era weights ─────────────────────────────────────────────────────
# More recent = more relevant to 2026 car physics
YEAR_WEIGHTS = {
    2018: 0.40,
    2019: 0.45,
    2020: 0.50,
    2021: 0.55,
    2022: 0.75,   # ground effect era begins
    2023: 0.85,
    2024: 1.00,
    2025: 1.20,   # most recent full season
    2026: 1.50,   # current season — highest weight
}


# ── ELO computation ────────────────────────────────────────────────────────────

def compute_elo_series(db: pd.DataFrame, K: float = 20.0) -> pd.DataFrame:
    """
    Computes pre-race ELO for every driver at every round.
    Uses Finish for completed races, skips mechanical DNFs in ELO update.
    """
    db   = db.sort_values(["Year","Round","Finish"]).reset_index(drop=True)
    elos = {}
    elo_records = []

    for _, row in db.iterrows():
        drv = row["Driver"]
        if drv not in elos:
            elos[drv] = 1500.0

    for (year, rnd), race_df in db.groupby(["Year","Round"]):
        # Record pre-race ELO
        for _, row in race_df.iterrows():
            elo_records.append({
                "Year"  : year,
                "Round" : rnd,
                "Driver": row["Driver"],
                "Elo"   : elos.get(row["Driver"], 1500.0)
            })

        # Update ELO — skip DNF drivers from update
        clean = race_df[race_df["DNF_Type"] != "mechanical"].copy()
        if len(clean) < 2:
            continue

        drvs = clean["Driver"].tolist()
        pos  = dict(zip(clean["Driver"], clean["Finish"]))
        n    = len(drvs)

        new_elos = elos.copy()
        for d1 in drvs:
            if d1 not in pos or pd.isna(pos[d1]):
                continue
            delta = 0.0
            for d2 in drvs:
                if d1 == d2 or d2 not in pos or pd.isna(pos[d2]):
                    continue
                exp   = 1 / (1 + 10 ** ((elos.get(d2,1500) - elos.get(d1,1500)) / 400))
                act   = 1.0 if pos[d1] < pos[d2] else 0.0
                delta += K * (act - exp)
            new_elos[d1] = elos.get(d1, 1500.0) + delta / (n - 1)
        elos = new_elos

    return pd.DataFrame(elo_records)


# ── DNF imputation ─────────────────────────────────────────────────────────────

def impute_dnf_positions(db: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces mechanical DNF finishing positions with expected finish
    based on grid position and driver's season median.
    """
    db = db.copy()
    mechanical = db["DNF_Type"] == "mechanical"

    # Driver season median on clean races
    clean    = db[~mechanical & (db["Finished"] == 1)]
    drv_med  = clean.groupby("Driver")["Finish"].median().to_dict()
    seas_med = clean.groupby(["Year","Driver"])["Finish"].median()

    def _impute(row):
        if not mechanical.loc[row.name]:
            return row["Finish"]
        drv  = row["Driver"]
        year = row["Year"]
        grid = row["Grid"]

        # Try season-specific median first, then career median
        med = seas_med.get((year, drv), drv_med.get(drv, 10.0))

        if pd.notna(grid) and grid > 0:
            return float(np.clip(0.60 * grid + 0.40 * med, 1, 20))
        return float(np.clip(med, 1, 20))

    db["Finish_Clean"] = db.apply(_impute, axis=1)
    return db


# ── rolling features ───────────────────────────────────────────────────────────

def build_rolling_features(db: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling features per driver, respecting chronological order.
    All features are computed BEFORE the race they apply to (no leakage).
    """
    db = db.sort_values(["Driver","Year","Round"]).reset_index(drop=True)

    N_FIELD = 20.0
    RW      = [0.50, 0.25, 0.15, 0.10]

    def _pos_score(p):
        return float(np.clip(1.0 - (p - 1) / (N_FIELD - 1), 0.0, 1.0))

    def _momentum(series):
        vals = series.values.astype(float)
        scores = []
        for i in range(len(vals)):
            start  = max(0, i - 3)
            window = vals[start:i+1][::-1]
            w      = RW[:len(window)]
            w      = [x / sum(w) for x in w]
            scores.append(sum(_pos_score(p) * wt
                              for p, wt in zip(window, w)))
        return pd.Series(scores, index=series.index)

    db["Momentum"] = db.groupby("Driver")["Finish_Clean"].transform(_momentum)

    # Qualifying gap EMA (span=4, pre-race)
    db["Quali_Gap_EMA"] = db.groupby("Driver")["Quali_Gap_s"].transform(
        lambda x: x.ewm(span=4, adjust=False).mean())

    # Points EMA (team performance)
    db["Points_EMA"] = db.groupby(["Year","Team"])["Points"].transform(
        lambda x: x.ewm(span=4, adjust=False).mean())

    # Reliability (Bayesian, expanding)
    def _reliability(group):
        fin   = (group["Finished"] == 1).astype(float)
        mech  = (group["DNF_Type"] == "mechanical").astype(float)
        prior_n = 3.0; prior_r = 0.85
        eff_fin = fin + mech * 0.5
        rel     = []
        for i in range(len(fin)):
            n   = i + prior_n
            f   = eff_fin.iloc[:i].sum() + prior_n * prior_r
            rel.append(f / n)
        return pd.Series(rel, index=group.index)

    db["Reliability"] = db.groupby("Driver").apply(_reliability)\
                          .reset_index(0, drop=True)

    # Skill delta (driver vs team, excluding mech DNF rounds)
    mech_in_team = db.groupby(["Year","Round","Team"])["DNF_Type"]\
                     .transform(lambda x: (x == "mechanical").any())
    team_avg     = db.groupby(["Year","Round","Team"])["Finish_Clean"]\
                     .transform("mean")
    db["Skill_Raw"]   = team_avg - db["Finish_Clean"]
    db.loc[mech_in_team, "Skill_Raw"] = 0.0
    db["Skill_Delta"] = db.groupby("Driver")["Skill_Raw"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean())

    # Grid position percentile (how far up the grid vs field)
    db["Grid_Pct"] = db.groupby(["Year","Round"])["Grid"].rank(pct=True,
                                                                ascending=False)

    return db


# ── feature matrix ─────────────────────────────────────────────────────────────

def build_feature_matrix(db: pd.DataFrame,
                          elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges all features into a single matrix ready for training.
    Target variable: Finish_Clean (1-20).
    """
    db = db.merge(elo_df, on=["Year","Round","Driver"], how="left")

    # Encode circuit type
    ct_map = {"street": 2, "balanced": 1, "power": 0}
    db["Circuit_Code"] = db["Circuit_Type"].map(ct_map).fillna(1)

    # Normalise ELO within each race (relative ELO matters more than absolute)
    db["Elo_Norm"] = db.groupby(["Year","Round"])["Elo"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9))

    # Season progress (round / total rounds)
    total_rounds = db.groupby("Year")["Round"].transform("max")
    db["Season_Progress"] = db["Round"] / total_rounds

    features = [
        # Race context
        "Circuit_Code",      # 0=power, 1=balanced, 2=street
        "Reg_Era",           # 0=2018-21, 1-3=newer eras
        "Season_Progress",   # 0-1, early vs late season
        "Grid",              # starting grid position
        "Grid_Pct",          # grid position percentile

        # Driver form
        "Momentum",          # recency-weighted position score
        "Elo_Norm",          # normalised ELO vs this race field
        "Skill_Delta",       # above/below teammate
        "Reliability",       # Bayesian finish rate

        # Pace
        "Quali_Gap_EMA",     # gap to pole EMA (season avg quali pace)
        "Points_EMA",        # team/driver points momentum
    ]

    # Drop rows with NaN in critical features
    critical = ["Grid","Momentum","Elo_Norm","Quali_Gap_EMA","Finish_Clean"]
    db = db.dropna(subset=critical)

    # Fill remaining NaNs with sensible defaults
    db["Skill_Delta"]    = db["Skill_Delta"].fillna(0.0)
    db["Reliability"]    = db["Reliability"].fillna(0.80)
    db["Points_EMA"]     = db["Points_EMA"].fillna(0.0)
    db["Grid_Pct"]       = db["Grid_Pct"].fillna(0.5)

    return db, features


# ── training ───────────────────────────────────────────────────────────────────

def train_model(db: pd.DataFrame, features: list) -> object:
    """
    Trains LightGBM to predict finishing position.
    Falls back to XGBoost if LightGBM not installed.
    """
    X = db[features].values.astype(float)
    y = db["Finish_Clean"].values.astype(float)
    w = db["Year"].map(YEAR_WEIGHTS).fillna(0.5).values

    # Train/validation split — use last 2 seasons as validation
    val_mask  = db["Year"] >= (CURRENT_YEAR - 2)
    X_train, X_val = X[~val_mask], X[val_mask]
    y_train, y_val = y[~val_mask], y[val_mask]
    w_train        = w[~val_mask]

    print(f"  Train: {len(X_train)} rows | Val: {len(X_val)} rows")

    try:
        import lightgbm as lgb

        dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
        dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain)

        params = {
            "objective"       : "regression",
            "metric"          : "rmse",
            "num_leaves"      : 31,
            "learning_rate"   : 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq"    : 5,
            "min_child_samples": 10,
            "verbose"         : -1,
            "n_jobs"          : -1,
        }

        callbacks = [lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(period=-1)]

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(((val_pred - y_val)**2).mean())
        print(f"  LightGBM RMSE (validation): {val_rmse:.3f} positions")

        # Feature importance
        imp_df = pd.DataFrame({
            "Feature"   : features,
            "Importance": model.feature_importance(importance_type="gain"),
        }).sort_values("Importance", ascending=False)

        return model, imp_df, "lightgbm", val_rmse

    except ImportError:
        print("  LightGBM not found — falling back to XGBoost")

        try:
            import xgboost as xgb

            model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            val_pred = model.predict(X_val)
            val_rmse = np.sqrt(((val_pred - y_val)**2).mean())
            print(f"  XGBoost RMSE (validation): {val_rmse:.3f} positions")

            imp_df = pd.DataFrame({
                "Feature"   : features,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)

            return model, imp_df, "xgboost", val_rmse

        except ImportError:
            raise ImportError(
                "Neither LightGBM nor XGBoost found. "
                "Install with: pip install lightgbm  OR  pip install xgboost"
            )


# ── main ───────────────────────────────────────────────────────────────────────

def train():
    print(f"\n╔══ APEX TRAIN — {datetime.now().strftime('%Y-%m-%d %H:%M')} ══╗")

    # 1. Load historical data
    if not os.path.exists(HIST_PATH):
        print(f"║  ERROR: {HIST_PATH} not found.")
        print("║  Run: python apex_historical.py first.")
        return

    hist = pd.read_csv(HIST_PATH)
    print(f"║  Historical data: {len(hist)} rows | "
          f"Years: {hist['Year'].min()}–{hist['Year'].max()}")

    # 2. Load current season data (if available)
    if os.path.exists(CURR_DB_PATH):
        curr = pd.read_csv(CURR_DB_PATH)
        # Rename columns to match historical schema
        curr = curr.rename(columns={
            "Round" : "Round",
            "Finish": "Finish",
            "Grid"  : "Grid",
        })
        curr["Year"]         = CURRENT_YEAR
        curr["Reg_Era"]      = 3
        curr["Circuit_Type"] = "balanced"  # will be overridden if known
        curr["DNF_Type"]     = curr["Status"].apply(_dnf_type_from_status)

        # Align columns
        for col in hist.columns:
            if col not in curr.columns:
                curr[col] = np.nan

        combined = pd.concat([hist, curr[hist.columns]], ignore_index=True)
        print(f"║  + Current season: {len(curr)} rows → "
              f"Total: {len(combined)} rows")
    else:
        combined = hist
        print("║  No current season data found — training on historical only")

    # 3. Impute DNFs
    print("\n║  [ Imputing mechanical DNF positions... ]")
    combined = impute_dnf_positions(combined)

    # 4. Rolling features
    print("║  [ Building rolling features... ]")
    combined = build_rolling_features(combined)

    # 5. ELO
    print("║  [ Computing ELO series... ]")
    elo_df = compute_elo_series(combined)

    # 6. Feature matrix
    print("║  [ Assembling feature matrix... ]")
    combined, features = build_feature_matrix(combined, elo_df)
    print(f"║  Features: {features}")
    print(f"║  Training rows: {len(combined)}")

    # 7. Train
    print("\n║  [ Training model... ]")
    model, imp_df, model_type, val_rmse = train_model(combined, features)

    # 8. Save
    os.makedirs("f1_data_historical", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model"     : model,
            "features"  : features,
            "model_type": model_type,
        }, f)

    imp_df.to_csv(FEAT_PATH, index=False)

    meta = {
        "trained_at"   : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_type"   : model_type,
        "val_rmse"     : round(val_rmse, 4),
        "n_train_rows" : len(combined),
        "years"        : f"{combined['Year'].min()}–{combined['Year'].max()}",
        "features"     : features,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n║  ✓ Model saved: {MODEL_PATH}")
    print(f"║  ✓ Metadata: {META_PATH}")
    print(f"\n║  Top 5 features by importance:")
    for _, row in imp_df.head(5).iterrows():
        bar = "█" * int(row["Importance"] / imp_df["Importance"].max() * 20)
        print(f"║    {row['Feature']:<20} {bar}")

    print(f"\n║  Validation RMSE: {val_rmse:.3f} positions")
    print(f"║  (Baseline random: ~5.7 | Good model: <3.5 | Excellent: <2.5)")
    print("╚══════════════════════════════╝\n")

    return model, features


def _dnf_type_from_status(status: str) -> str:
    if pd.isna(status):
        return "none"
    s = str(status).lower()
    if any(w in s for w in ["retired","did not start","power","gearbox",
                             "engine","hydraulic","electrical","mechanical",
                             "brake","suspension","exhaust","fuel"]):
        return "mechanical"
    if any(w in s for w in ["accident","collision","spin","damage"]):
        return "driver"
    if "finished" in s or "lapped" in s or "+" in s:
        return "none"
    return "mechanical"


if __name__ == "__main__":
    train()