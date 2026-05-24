"""
APEX PREDATOR — apex_predict.py
=================================
Unified autonomous prediction engine.

Architecture:
  Signal 1 — Historical circuit DNA   (always available from race 1)
  Signal 2 — Qualifying position      (strongest single-race predictor)
  Signal 3 — Current season form      (ELO + EMA, grows each race)
  Signal 4 — Car physics / aero fit   (track-weighted telemetry)
  Signal 5 — Reliability              (Bayesian DNF risk)

Weights shift automatically as the season progresses:
  Early (1-4 races):  History 40%, Quali 30%, Form 15%, Physics 10%, Rel 5%
  Mid   (5-10 races): History 20%, Quali 30%, Form 30%, Physics 15%, Rel 5%
  Late  (11+ races):  History 10%, Quali 25%, Form 40%, Physics 20%, Rel 5%

Single command: python apex_predict.py
Outputs:
  f1_data_YYYY/predictions.csv   — full grid prediction
  f1_data_YYYY/database.csv      — updated feature database (if new race)
"""

import fastf1
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from apex_ingest import ingest

warnings.filterwarnings("ignore")
np.random.seed(42)

YEAR      = datetime.now().year
DATA_DIR  = f"f1_data_{YEAR}"
DB_PATH   = f"{DATA_DIR}/database.csv"
PRED_PATH = f"{DATA_DIR}/predictions.csv"
CACHE_DIR = "f1_cache"

os.makedirs(DATA_DIR,  exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

POINTS_MAP = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
N_SIMS     = 50_000

# ── weight schedule ────────────────────────────────────────────────────────────
def _weights(n_rounds: int) -> dict:
    if n_rounds <= 4:
        return {"history": 0.40, "quali": 0.30, "form": 0.15,
                "physics": 0.10, "reliability": 0.05}
    elif n_rounds <= 10:
        return {"history": 0.20, "quali": 0.30, "form": 0.30,
                "physics": 0.15, "reliability": 0.05}
    else:
        return {"history": 0.10, "quali": 0.25, "form": 0.40,
                "physics": 0.20, "reliability": 0.05}


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 1 — HISTORICAL CIRCUIT DNA
# ══════════════════════════════════════════════════════════════════════════════

def _pos_to_score(pos: float) -> float:
    """P1=1.0, exponential decay. P10≈0.22, P20≈0.03"""
    if pd.isna(pos) or pos <= 0:
        return 0.0
    return float(np.exp(-(pos - 1) / 5.5))


def _normalise_team(t: str) -> str:
    return t.lower().replace(" f1 team","").replace(" racing","").strip()


def build_history_signal(next_race: str, current_drivers: pd.DataFrame) -> pd.Series:
    """
    Pulls last 3 years of results at this circuit.
    Returns a Series indexed by Driver abbreviation, values 0-1.
    """
    year_weights = {
        YEAR - 1: 0.50,
        YEAR - 2: 0.30,
        YEAR - 3: 0.20,
    }

    all_hist = []
    for yr, w in year_weights.items():
        try:
            s = fastf1.get_session(yr, next_race, "R")
            s.load(telemetry=False, weather=False, messages=False)
            res = s.results[["Abbreviation","TeamName","Position"]].copy()
            res.columns = ["Driver","Team","Position"]
            res["Position"] = pd.to_numeric(res["Position"], errors="coerce")
            res["Weight"]   = w
            res["Year"]     = yr
            all_hist.append(res.dropna(subset=["Position"]))
            print(f"  History {yr}: {len(res)} drivers")
        except Exception as e:
            print(f"  History {yr}: unavailable ({e})")

    drivers    = current_drivers["Driver"].tolist()
    teams      = dict(zip(current_drivers["Driver"], current_drivers["Team"]))
    scores     = {}

    if not all_hist:
        # No history at all — neutral 0.5 for everyone
        return pd.Series({d: 0.5 for d in drivers})

    hist_df     = pd.concat(all_hist, ignore_index=True)

    # Build team-level average for proxy
    team_avg    = hist_df.groupby("Team")["Position"].mean().to_dict()
    team_avg_n  = {_normalise_team(k): v for k, v in team_avg.items()}

    for drv in drivers:
        dh = hist_df[hist_df["Driver"] == drv]
        if not dh.empty:
            # Weighted average across years
            num = sum(_pos_to_score(r["Position"]) * r["Weight"]
                      for _, r in dh.iterrows())
            den = dh["Weight"].sum()
            scores[drv] = num / den
        else:
            # Team proxy with 40% discount
            team_key = _normalise_team(teams.get(drv, ""))
            match    = next(
                (v for k, v in team_avg_n.items()
                 if team_key in k or k in team_key), None)
            scores[drv] = _pos_to_score(match) * 0.60 if match else 0.3

    # Normalise to 0-1
    arr  = np.array(list(scores.values()))
    rng  = arr.max() - arr.min()
    if rng > 1e-6:
        for d in scores:
            scores[d] = (scores[d] - arr.min()) / rng
    else:
        for d in scores:
            scores[d] = 0.5

    return pd.Series(scores)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 2 — QUALIFYING POSITION
# ══════════════════════════════════════════════════════════════════════════════

def build_quali_signal(grid: pd.DataFrame) -> pd.Series:
    """
    Uses Quali_Gap_EMA (gap to pole in seconds, lower = better).
    Returns 0-1 score where 0.0 gap (pole) → 1.0, large gap → 0.0.
    """
    gap = grid["Quali_Gap_EMA"].values
    # Invert: smaller gap = higher score
    inv = gap.max() - gap
    rng = inv.max() - inv.min()
    if rng > 1e-6:
        norm = (inv - inv.min()) / rng
    else:
        norm = np.full(len(inv), 0.5)
    return pd.Series(norm, index=grid.index)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 3 — CURRENT SEASON FORM (ELO + EMA)
# ══════════════════════════════════════════════════════════════════════════════

def elo_update(ratings: dict, race_df: pd.DataFrame, K: float = 20.0) -> dict:
    new   = ratings.copy()
    drvs  = race_df["Driver"].tolist()
    pos   = dict(zip(race_df["Driver"], race_df["Finish"]))
    n     = len(drvs)
    for d1 in drvs:
        delta = 0.0
        for d2 in drvs:
            if d1 == d2: continue
            exp   = 1 / (1 + 10 ** ((ratings[d2] - ratings[d1]) / 400))
            act   = 1.0 if pos[d1] < pos[d2] else 0.0
            delta += K * (act - exp)
        new[d1] = ratings[d1] + delta / (n - 1)
    return new


def build_form_features(db: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-round ELO, Form_EMA, Quali_Gap_EMA to the database.
    Returns enriched DataFrame.
    """
    db   = db.sort_values(["Round", "Finish"]).reset_index(drop=True)
    elos = {d: 1500.0 for d in db["Driver"].unique()}
    elo_records = []

    for rnd in sorted(db["Round"].unique()):
        rdf = db[db["Round"] == rnd]
        for _, row in rdf.iterrows():
            elo_records.append({
                "Round" : rnd,
                "Driver": row["Driver"],
                "Elo"   : elos[row["Driver"]]
            })
        elos = elo_update(elos, rdf)

    elo_df = pd.DataFrame(elo_records)
    db     = db.merge(elo_df, on=["Round","Driver"], how="left")

    db = db.sort_values(["Driver","Round"])

    # EMA span=5 for form, expanding mean for telemetry
    db["Form_EMA"]      = db.groupby("Driver")["Finish"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean())
    db["Points_EMA"]    = db.groupby("Driver")["Points"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean())
    db["Quali_Gap_EMA"] = db.groupby("Driver")["Quali_Gap_s"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean())
    db["Top_Speed_SMA"] = db.groupby("Driver")["Top_Speed"].expanding().mean()\
                            .reset_index(0, drop=True)
    db["Corner_SMA"]    = db.groupby("Driver")["Corner_Speed"].expanding().mean()\
                            .reset_index(0, drop=True)

    return db


def build_form_signal(grid: pd.DataFrame) -> pd.Series:
    """ELO + Form_EMA combined, normalised 0-1. Higher = better."""
    elo_min = grid["Elo"].min(); elo_max = grid["Elo"].max()
    elo_sc  = (grid["Elo"] - elo_min) / (elo_max - elo_min + 1e-9)

    # Form_EMA: lower finish = better, so invert
    frm_min = grid["Form_EMA"].min(); frm_max = grid["Form_EMA"].max()
    frm_sc  = 1.0 - (grid["Form_EMA"] - frm_min) / (frm_max - frm_min + 1e-9)

    combined = 0.60 * frm_sc + 0.40 * elo_sc
    return combined.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 4 — CAR PHYSICS / AERO FIT
# ══════════════════════════════════════════════════════════════════════════════

def _track_dna(next_race: str) -> dict:
    """
    Pulls prior-year qualifying telemetry for the next circuit.
    Returns track weights and profile label.
    """
    default = {"w_speed": 1.0, "w_corner": 1.0, "profile": "Balanced (fallback)"}
    try:
        s   = fastf1.get_session(YEAR - 1, next_race, "Q")
        s.load(telemetry=True, weather=False, messages=False)
        tel = s.laps.pick_fastest().get_telemetry()
        if tel is None or tel.empty:
            return default

        spd            = tel["Speed"]
        thr            = tel["Throttle"]
        ft_pct         = (thr == 100).sum() / len(thr)
        avg_spd        = spd.mean()
        corner_density = (spd < spd.max() * 0.85).sum() / len(spd)

        print(f"  Track DNA → FT%: {ft_pct*100:.1f}% | "
              f"Avg: {avg_spd:.1f} km/h | Corners: {corner_density*100:.1f}%")

        is_power  = ft_pct > 0.68 or avg_spd > 230
        is_twisty = corner_density > 0.50 or avg_spd < 195

        if is_power and not is_twisty:
            return {"w_speed": 1.25, "w_corner": 0.75, "profile": "Power circuit"}
        elif is_twisty and not is_power:
            return {"w_speed": 0.75, "w_corner": 1.25, "profile": "Downforce circuit"}
        else:
            return {"w_speed": 1.0,  "w_corner": 1.0,  "profile": "Balanced circuit"}
    except Exception as e:
        print(f"  Track DNA unavailable: {e}")
        return default


def build_physics_signal(grid: pd.DataFrame, track: dict) -> pd.Series:
    """
    Team-level aero Z-scores weighted by track profile.
    Returns 0-1 score per driver.
    """
    # Z-score telemetry SMAs
    for col, zcol in [("Top_Speed_SMA","Spd_Z"), ("Corner_SMA","Cor_Z")]:
        mu  = grid[col].mean()
        std = grid[col].std() + 1e-9
        grid[zcol] = np.clip((grid[col] - mu) / std, -2.5, 2.5)

    # Team homogeneity: teammates share chassis Z
    grid["Spd_Z"] = grid.groupby("Team")["Spd_Z"].transform("mean")
    grid["Cor_Z"] = grid.groupby("Team")["Cor_Z"].transform("mean")

    aero = (grid["Spd_Z"] * track["w_speed"] +
            grid["Cor_Z"] * track["w_corner"])

    # Normalise to 0-1
    rng = aero.max() - aero.min()
    if rng > 1e-6:
        return ((aero - aero.min()) / rng).reset_index(drop=True)
    return pd.Series(np.full(len(aero), 0.5))


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 5 — RELIABILITY
# ══════════════════════════════════════════════════════════════════════════════

def build_reliability_signal(db: pd.DataFrame, grid: pd.DataFrame) -> pd.Series:
    """Bayesian finish rate with weak prior of 80%."""
    prior_n   = 3.0
    prior_r   = 0.80
    drv_n     = db.groupby("Driver").size()
    drv_fin   = db[db["Finished"] == 1].groupby("Driver").size()
    rel       = (drv_fin.fillna(0) + prior_n * prior_r) / (drv_n + prior_n)
    mapped    = grid["Driver"].map(rel).fillna(prior_r)
    # Normalise
    rng       = mapped.max() - mapped.min()
    if rng > 1e-6:
        return ((mapped - mapped.min()) / rng).reset_index(drop=True)
    return pd.Series(np.full(len(mapped), 0.5))


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def _temperature(scores: np.ndarray, n_rounds: int) -> float:
    """
    Std-based temperature selection.
    Early season: wider (more uncertainty).
    Late season: tighter (data-driven).
    """
    std     = scores.std()
    scaling = {True: 1.2, False: 1.0}[n_rounds <= 6]
    if n_rounds > 10:
        scaling = 0.85
    return float(np.clip(std * scaling, 0.15, 2.0))


def simulate(drivers: list, probs: np.ndarray) -> pd.DataFrame:
    n        = len(drivers)
    wins     = np.zeros(n)
    podiums  = np.zeros(n)
    top6     = np.zeros(n)
    top10    = np.zeros(n)
    pts_acc  = np.zeros(n)

    gumbel   = np.random.gumbel(0, 1, (N_SIMS, n))
    perturb  = np.log(probs + 1e-12) + gumbel
    rankings = np.argsort(-perturb, axis=1)

    for pos in range(1, n + 1):
        idx = rankings[:, pos - 1]
        if pos == 1:  np.add.at(wins,    idx, 1)
        if pos <= 3:  np.add.at(podiums, idx, 1)
        if pos <= 6:  np.add.at(top6,    idx, 1)
        if pos <= 10: np.add.at(top10,   idx, 1)
        pts = POINTS_MAP.get(pos, 0)
        if pts:       np.add.at(pts_acc, idx, pts)

    p  = wins / N_SIMS
    ci = 1.96 * np.sqrt(p * (1 - p) / N_SIMS) * 100

    rows = []
    for i, d in enumerate(drivers):
        rows.append({
            "Driver"   : d,
            "Win"      : round(wins[i]    / N_SIMS * 100, 2),
            "Win_CI"   : round(ci[i],                      2),
            "Podium"   : round(podiums[i] / N_SIMS * 100, 2),
            "Top6"     : round(top6[i]    / N_SIMS * 100, 2),
            "Top10"    : round(top10[i]   / N_SIMS * 100, 2),
            "Points_EV": round(pts_acc[i] / N_SIMS,       2),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def predict():
    print(f"\n╔══ APEX PREDICT — {YEAR} SEASON ══╗")

    # 1. Ingest any new races
    db = ingest()
    if db.empty:
        print("║  No race data available yet.")
        return

    # 2. Identify next race
    schedule = fastf1.get_event_schedule(YEAR)
    upcoming = schedule[schedule["EventDate"] > pd.Timestamp.now()]
    if upcoming.empty:
        print("║  Season complete.")
        return
    next_event = upcoming.iloc[0]
    next_name  = next_event["EventName"]
    n_rounds   = int(db["Round"].max())
    print(f"║  Rounds completed: {n_rounds} | Next race: {next_name}\n")

    # 3. Build features
    print("║  [ Building features... ]")
    db = build_form_features(db)

    # 4. Current grid = latest round snapshot
    latest = db[db["Round"] == db["Round"].max()].copy().reset_index(drop=True)
    drivers = latest["Driver"].tolist()
    teams   = dict(zip(latest["Driver"], latest["Team"]))

    # 5. Track DNA
    print(f"\n║  [ Track DNA: {next_name} ]")
    track = _track_dna(next_name)
    print(f"║    Profile: {track['profile']}")

    # 6. Build all signals
    print(f"\n║  [ Building signals (n_rounds={n_rounds}) ]")

    print("║    Signal 1: Historical circuit DNA...")
    hist_signal = build_history_signal(
        next_name,
        pd.DataFrame({"Driver": drivers, "Team": [teams[d] for d in drivers]})
    )

    print("║    Signal 2: Qualifying form...")
    quali_signal = build_quali_signal(latest)

    print("║    Signal 3: Current season form...")
    form_signal = build_form_signal(latest)

    print("║    Signal 4: Car physics...")
    phys_signal = build_physics_signal(latest.copy(), track)

    print("║    Signal 5: Reliability...")
    rel_signal = build_reliability_signal(db, latest)

    # 7. Weighted composite score
    W = _weights(n_rounds)
    print(f"\n║  [ Weight schedule: "
          f"History {W['history']*100:.0f}% | "
          f"Quali {W['quali']*100:.0f}% | "
          f"Form {W['form']*100:.0f}% | "
          f"Physics {W['physics']*100:.0f}% | "
          f"Rel {W['reliability']*100:.0f}% ]")

    # Map history signal (indexed by driver) to grid order
    hist_arr = np.array([hist_signal.get(d, 0.5) for d in drivers])

    score = (
        W["history"]     * hist_arr           +
        W["quali"]       * quali_signal.values +
        W["form"]        * form_signal.values  +
        W["physics"]     * phys_signal.values  +
        W["reliability"] * rel_signal.values
    )

    # 8. Temperature and probabilities
    temp  = _temperature(score, n_rounds)
    shifted = (score - score.max()) / temp
    exp_s   = np.exp(shifted)
    probs   = exp_s / exp_s.sum()

    print(f"║  [ Temperature: {temp:.3f} | "
          f"Score range: {score.min():.3f}–{score.max():.3f} | "
          f"Max prob: {probs.max()*100:.1f}% ]\n")

    # 9. Pre-simulation ranking
    pre = pd.DataFrame({
        "Driver" : drivers,
        "Team"   : [teams[d] for d in drivers],
        "Score"  : score,
        "RawProb": probs * 100,
        "History": hist_arr,
        "Form"   : form_signal.values,
        "Quali"  : quali_signal.values,
        "Physics": phys_signal.values,
    }).sort_values("Score", ascending=False)

    print("║  [ Pre-simulation signal breakdown (top 12) ]")
    print(f"║  {'Driver':<7}{'Team':<20}{'Score':>7}{'Raw%':>7}"
          f"{'Hist':>7}{'Form':>7}{'Quali':>7}{'Phys':>7}")
    print("║  " + "─" * 68)
    for _, r in pre.head(12).iterrows():
        print(f"║  {r['Driver']:<7}{r['Team']:<20}"
              f"{r['Score']:>7.3f}{r['RawProb']:>6.1f}%"
              f"{r['History']:>7.3f}{r['Form']:>7.3f}"
              f"{r['Quali']:>7.3f}{r['Physics']:>7.3f}")

    # 10. Monte Carlo
    print(f"\n║  [ Running {N_SIMS:,} simulations... ]")
    sim_df = simulate(drivers, probs)
    sim_df["Team"] = sim_df["Driver"].map(teams)
    sim_df = sim_df.sort_values("Win", ascending=False).reset_index(drop=True)

    # 11. Print results
    print(f"\n╠══ PREDICTION: {next_name.upper()} ══╣")
    print(f"║  {'Pos':<4}{'Driver':<7}{'Team':<22}"
          f"{'Win%':>6}{'±':>5}{'Podium%':>9}{'Top10%':>8}{'Pts EV':>8}")
    print("║  " + "─" * 68)
    for i, row in sim_df.head(15).iterrows():
        print(f"║  {i+1:<4}{row['Driver']:<7}{row['Team']:<22}"
              f"{row['Win']:>6.1f}{row['Win_CI']:>5.1f}"
              f"{row['Podium']:>9.1f}{row['Top10']:>8.1f}"
              f"{row['Points_EV']:>8.2f}")

    # 12. Head-to-head matrix (top 6)
    top6_drvs = sim_df.head(6)["Driver"].tolist()
    print(f"\n║  [ Head-to-head Win% edge ]")
    print(f"║  {'':>7}", end="")
    for d in top6_drvs:
        print(f"{d:>8}", end="")
    print()
    for d1 in top6_drvs:
        w1 = sim_df[sim_df["Driver"]==d1]["Win"].values[0]
        print(f"║  {d1:>7}", end="")
        for d2 in top6_drvs:
            if d1 == d2:
                print(f"{'—':>8}", end="")
            else:
                w2   = sim_df[sim_df["Driver"]==d2]["Win"].values[0]
                edge = w1 / (w1 + w2) * 100 if (w1+w2) > 0 else 50
                print(f"{edge:>7.0f}%", end="")
        print()

    # 13. Save
    sim_df["Next_Race"]  = next_name
    sim_df["Predicted_At"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    sim_df["N_Rounds"]   = n_rounds
    sim_df.to_csv(PRED_PATH, index=False)
    print(f"\n║  ✓ Predictions saved: {PRED_PATH}")
    print("╚══════════════════════════════╝\n")

    return sim_df


if __name__ == "__main__":
    predict()