"""
PITWALL — apex_train.py
================================
Fully autonomous feature engineering + LightGBM training.

Zero hardcoding:
  - Current grid auto-detected from live database
  - Historical filtering by race count (>=5 per season), not driver names
  - Team names normalised dynamically
  - Rookies handled via neutral priors
  - New tracks handled via circuit DNA

Input:  f1_data_historical/historical_database.csv
Output: f1_data_historical/apex_model.pkl
        f1_data_historical/feature_importance.csv
        f1_data_historical/model_metadata.json

Run after every race: python apex_train.py
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

HIST_PATH    = "f1_data_historical/historical_database.csv"
MODEL_PATH   = "f1_data_historical/apex_model.pkl"
META_PATH    = "f1_data_historical/model_metadata.json"
FEAT_PATH    = "f1_data_historical/feature_importance.csv"
CURRENT_YEAR = datetime.now().year
CURR_PATH    = f"f1_data_{CURRENT_YEAR}/database.csv"

YEAR_WEIGHTS = {
    2018: 0.40, 2019: 0.45, 2020: 0.50,
    2021: 0.55, 2022: 0.75, 2023: 0.85,
    2024: 1.00, 2025: 1.20, 2026: 1.50,
}


def _normalise_team(team: str) -> str:
    if pd.isna(team):
        return "Unknown"
    t = str(team).strip()
    for suffix in [" F1 Team", " Racing", " Grand Prix", " Motorsport"]:
        t = t.replace(suffix, "")
    rebrands = {
        "AlphaTauri" : "Racing Bulls",
        "Toro Rosso" : "Racing Bulls",
        "Alfa Romeo" : "Sauber",
        "Renault"    : "Alpine",
        "Racing Point": "Aston Martin",
        "Force India" : "Aston Martin",
        "Sahara Force India": "Aston Martin",
        "BWT Racing Point"  : "Aston Martin",
        "Haas F1"    : "Haas",
    }
    for old, new in rebrands.items():
        if old.lower() in t.lower():
            return new
    return t.strip()


def _dnf_type(status: str) -> str:
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


def load_and_clean() -> pd.DataFrame:
    frames = []

    if os.path.exists(HIST_PATH):
        hist = pd.read_csv(HIST_PATH)
        print(f"  Historical: {len(hist)} rows | "
              f"{hist['Year'].min()}–{hist['Year'].max()}")
        frames.append(hist)
    else:
        print(f"  WARNING: {HIST_PATH} not found")

    if os.path.exists(CURR_PATH):
        curr = pd.read_csv(CURR_PATH)
        curr["Year"]    = CURRENT_YEAR
        curr["Reg_Era"] = 3
        if "Circuit_Type" not in curr.columns:
            curr["Circuit_Type"] = "balanced"
        if "DNF_Type" not in curr.columns:
            curr["DNF_Type"] = curr["Status"].apply(_dnf_type)
        if "Finished" not in curr.columns:
            curr["Finished"] = curr["Status"].apply(
                lambda s: int(bool(pd.Series([s])
                              .str.contains(r"Finished|Lapped|\+\d",
                                            regex=True, na=False).iloc[0])))
        curr["Team"] = curr["Team"].apply(_normalise_team)
        for col in pd.read_csv(HIST_PATH, nrows=0).columns \
                   if os.path.exists(HIST_PATH) else []:
            if col not in curr.columns:
                curr[col] = np.nan
        frames.append(curr)
        print(f"  Current season: {len(curr)} rows")

    if not frames:
        raise FileNotFoundError("No data found. Run apex_historical.py first.")

    db = pd.concat(frames, ignore_index=True)
    db["Team"]    = db["Team"].apply(_normalise_team)
    db["DNF_Type"] = db["Status"].apply(_dnf_type) \
                     if "Status" in db.columns else "none"
    db["Finish"]  = pd.to_numeric(db["Finish"],  errors="coerce")
    db["Grid"]    = pd.to_numeric(db["Grid"],     errors="coerce")
    db["Points"]  = pd.to_numeric(db["Points"],   errors="coerce").fillna(0)
    db["Year"]    = pd.to_numeric(db["Year"],     errors="coerce")
    db["Round"]   = pd.to_numeric(db["Round"],    errors="coerce")

    # Filter replacement drivers (<5 races per season)
    race_counts = db.groupby(["Year","Driver"])["Round"].count().reset_index()
    race_counts.columns = ["Year","Driver","Race_Count"]
    db = db.merge(race_counts, on=["Year","Driver"], how="left")
    n_excl = len(db[db["Race_Count"] < 5][["Year","Driver"]].drop_duplicates())
    print(f"  Filtering {n_excl} replacement driver-seasons (<5 races)")
    db = db[db["Race_Count"] >= 5].drop(columns=["Race_Count"])
    db = db.dropna(subset=["Finish"])

    print(f"  Combined: {len(db)} rows | "
          f"{db['Driver'].nunique()} drivers | "
          f"{db['Year'].nunique()} seasons")
    return db.reset_index(drop=True)


def impute_dnf(db: pd.DataFrame) -> pd.DataFrame:
    db   = db.copy()
    mech = db["DNF_Type"] == "mechanical"
    seas_med   = db[~mech & (db["Finished"]==1)]\
                   .groupby(["Year","Driver"])["Finish"].median()
    career_med = db[~mech & (db["Finished"]==1)]\
                   .groupby("Driver")["Finish"].median()

    def _impute(row):
        if not mech.loc[row.name]:
            return row["Finish"]
        drv  = row["Driver"]; yr = row["Year"]; grid = row["Grid"]
        med  = seas_med.get((yr,drv), career_med.get(drv, 10.0))
        if pd.notna(grid) and grid > 0:
            return float(np.clip(0.60*grid + 0.40*med, 1, 20))
        return float(np.clip(med, 1, 20))

    db["Finish_Clean"]   = db.apply(_impute, axis=1)
    db["Mechanical_DNF"] = mech.astype(int)
    return db


def compute_elo(db: pd.DataFrame, K: float = 20.0) -> pd.DataFrame:
    db   = db.sort_values(["Year","Round","Finish_Clean"]).reset_index(drop=True)
    elos = {}
    elo_records = []
    prev_year   = None

    for (year, rnd), race_df in db.groupby(["Year","Round"]):
        if year != prev_year and prev_year is not None:
            for d in elos:
                elos[d] = 1500 + (elos[d] - 1500) * 0.90
        prev_year = year

        for drv in race_df["Driver"].unique():
            if drv not in elos:
                elos[drv] = 1500.0

        for _, row in race_df.iterrows():
            elo_records.append({
                "Year": year, "Round": rnd,
                "Driver": row["Driver"],
                "Elo": elos[row["Driver"]]
            })

        clean = race_df[race_df["DNF_Type"] != "mechanical"].copy()
        if len(clean) < 2:
            continue

        drvs = clean["Driver"].tolist()
        pos  = dict(zip(clean["Driver"], clean["Finish_Clean"]))
        n    = len(drvs)
        new_elos = elos.copy()

        for d1 in drvs:
            if pd.isna(pos.get(d1)): continue
            delta = 0.0
            for d2 in drvs:
                if d1==d2 or pd.isna(pos.get(d2)): continue
                exp   = 1/(1+10**((elos.get(d2,1500)-elos.get(d1,1500))/400))
                act   = 1.0 if pos[d1] < pos[d2] else 0.0
                delta += K*(act-exp)
            new_elos[d1] = elos.get(d1,1500.0) + delta/max(n-1,1)
        elos = new_elos

    return pd.DataFrame(elo_records)


def build_rolling_features(db: pd.DataFrame) -> pd.DataFrame:
    db   = db.sort_values(["Driver","Year","Round"]).reset_index(drop=True)
    RW   = [0.50, 0.25, 0.15, 0.10]

    def _pos_score(p):
        return float(np.clip(1.0-(p-1)/19.0, 0.0, 1.0))

    def _momentum(series):
        vals = series.values.astype(float)
        scores = []
        for i in range(len(vals)):
            start  = max(0, i-3)
            window = vals[start:i+1][::-1]
            w      = RW[:len(window)]
            w      = [x/sum(w) for x in w]
            scores.append(sum(_pos_score(p)*wt for p,wt in zip(window,w)))
        return pd.Series(scores, index=series.index)

    db["Momentum"] = db.groupby("Driver")["Finish_Clean"].transform(_momentum)

    db["Quali_Gap_EMA"] = db.groupby("Driver")["Quali_Gap_s"].transform(
        lambda x: x.ewm(span=4, adjust=False).mean())

    db["Team_Pts_EMA"] = db.groupby(["Year","Team"])["Points"].transform(
        lambda x: x.ewm(span=4, adjust=False).mean())

    def _reliability(group):
        fin    = (group["Finished"]==1).astype(float)
        mech   = (group["DNF_Type"]=="mechanical").astype(float)
        prior_n = 3.0; prior_r = 0.85
        eff_fin = fin + mech*0.5
        rel = []
        for i in range(len(fin)):
            n = i + prior_n
            f = eff_fin.iloc[:i].sum() + prior_n*prior_r
            rel.append(f/n)
        return pd.Series(rel, index=group.index)

    db["Reliability"] = db.groupby("Driver").apply(_reliability)\
                          .reset_index(0, drop=True)

    mech_in_team = db.groupby(["Year","Round","Team"])["DNF_Type"]\
                     .transform(lambda x: (x=="mechanical").any())
    team_size    = db.groupby(["Year","Round","Team"])["Driver"]\
                     .transform("count")
    team_avg     = db.groupby(["Year","Round","Team"])["Finish_Clean"]\
                     .transform("mean")
    db["Skill_Raw"] = team_avg - db["Finish_Clean"]
    db.loc[mech_in_team | (team_size<2), "Skill_Raw"] = 0.0
    db["Skill_Delta"] = db.groupby("Driver")["Skill_Raw"].transform(
        lambda x: x.ewm(span=10, adjust=False).mean())
    db["Skill_Delta"] = db["Skill_Delta"].clip(-3.0, 3.0)

    db["Grid_Pct"] = db.groupby(["Year","Round"])["Grid"]\
                       .rank(pct=True, ascending=False).fillna(0.5)
    return db


def build_feature_matrix(db: pd.DataFrame,
                          elo_df: pd.DataFrame) -> tuple:
    db = db.merge(elo_df, on=["Year","Round","Driver"], how="left")

    ct_map = {"street":2, "balanced":1, "power":0}
    db["Circuit_Code"] = db["Circuit_Type"].map(ct_map).fillna(1).astype(int)

    db["Elo_Norm"] = db.groupby(["Year","Round"])["Elo"].transform(
        lambda x: (x-x.mean())/(x.std()+1e-9))

    total_rnds = db.groupby("Year")["Round"].transform("max")
    db["Season_Progress"] = db["Round"] / total_rnds

    yr_min = db["Year"].min(); yr_max = db["Year"].max()
    db["Year_Norm"] = (db["Year"]-yr_min)/(yr_max-yr_min+1e-9)

    features = [
        "Circuit_Code", "Season_Progress", "Year_Norm",
        "Grid", "Grid_Pct", "Momentum", "Elo_Norm",
        "Skill_Delta", "Reliability", "Quali_Gap_EMA", "Team_Pts_EMA",
    ]

    critical = ["Grid","Momentum","Elo_Norm","Quali_Gap_EMA","Finish_Clean"]
    db = db.dropna(subset=critical)
    db["Skill_Delta"]  = db["Skill_Delta"].fillna(0.0)
    db["Reliability"]  = db["Reliability"].fillna(0.80)
    db["Team_Pts_EMA"] = db["Team_Pts_EMA"].fillna(0.0)
    db["Grid_Pct"]     = db["Grid_Pct"].fillna(0.5)
    db["Year_Norm"]    = db["Year_Norm"].fillna(0.5)

    return db, features


def train_model(db: pd.DataFrame, features: list) -> tuple:
    X = db[features].values.astype(float)
    y = db["Finish_Clean"].values.astype(float)
    w = db["Year"].map(YEAR_WEIGHTS).fillna(0.5).values

    val_mask  = db["Year"] >= (CURRENT_YEAR - 2)
    X_tr, X_v = X[~val_mask], X[val_mask]
    y_tr, y_v = y[~val_mask], y[val_mask]
    w_tr      = w[~val_mask]

    print(f"  Train: {len(X_tr)} rows | Val: {len(X_v)} rows")

    try:
        import lightgbm as lgb

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                             feature_name=features)
        dval   = lgb.Dataset(X_v,  label=y_v,  reference=dtrain)

        params = {
            "objective":"regression", "metric":"rmse",
            "num_leaves":63, "learning_rate":0.03,
            "feature_fraction":0.80, "bagging_fraction":0.80,
            "bagging_freq":5, "min_child_samples":15,
            "lambda_l1":0.1, "lambda_l2":0.1,
            "verbose":-1, "n_jobs":-1,
        }

        cbs = [lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(period=-1)]

        model = lgb.train(params, dtrain, num_boost_round=1000,
                          valid_sets=[dval], callbacks=cbs)

        val_pred = model.predict(X_v)
        rmse     = np.sqrt(((val_pred-y_v)**2).mean())
        print(f"  LightGBM RMSE: {rmse:.3f} positions")

        imp = pd.DataFrame({
            "Feature"   : features,
            "Importance": model.feature_importance(importance_type="gain"),
        }).sort_values("Importance", ascending=False)

        return model, imp, "lightgbm", rmse

    except ImportError:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            tree_method="hist", random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_v,y_v)], verbose=False)
        val_pred = model.predict(X_v)
        rmse     = np.sqrt(((val_pred-y_v)**2).mean())
        print(f"  XGBoost RMSE: {rmse:.3f} positions")
        imp = pd.DataFrame({
            "Feature"   : features,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        return model, imp, "xgboost", rmse


def train():
    print(f"\n╔══ APEX TRAIN — {datetime.now().strftime('%Y-%m-%d %H:%M')} ══╗")

    print("\n║  [ Step 1: Load & clean ]")
    db = load_and_clean()

    print("\n║  [ Step 2: Impute mechanical DNFs ]")
    db = impute_dnf(db)

    print("\n║  [ Step 3: ELO series ]")
    elo_df = compute_elo(db)

    print("\n║  [ Step 4: Rolling features ]")
    db = build_rolling_features(db)

    print("\n║  [ Step 5: Feature matrix ]")
    db, features = build_feature_matrix(db, elo_df)
    print(f"  Matrix: {len(db)} rows × {len(features)} features")

    print("\n║  [ Step 6: Train ]")
    model, imp_df, model_type, rmse = train_model(db, features)

    os.makedirs("f1_data_historical", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model":model, "features":features,
                     "model_type":model_type}, f)

    imp_df.to_csv(FEAT_PATH, index=False)

    meta = {
        "trained_at"  : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_type"  : model_type,
        "val_rmse"    : round(rmse, 4),
        "n_rows"      : len(db),
        "n_drivers"   : db["Driver"].nunique(),
        "years"       : f"{db['Year'].min()}–{db['Year'].max()}",
        "features"    : features,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n║  Model saved: {MODEL_PATH}")
    print(f"║  RMSE: {rmse:.3f} positions")
    print(f"\n║  Feature importance:")
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["Importance"]/imp_df["Importance"].max()*25)
        print(f"║    {row['Feature']:<20} {bar}")

    print("\n╚══════════════════════════════╝\n")
    return model, features


if __name__ == "__main__":
    train()