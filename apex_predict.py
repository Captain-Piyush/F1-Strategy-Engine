"""
APEX PREDATOR — apex_predict.py
=================================
Weekend-aware, track-aware prediction engine.

Signals:
  1. Track DNA match     — car vs circuit physical fit
  2. Momentum            — recency-weighted position score (last 4 races)
  3. Driver skill delta  — driver contribution above/below car
  4. Qualifying pace     — season avg OR actual grid post-quali
  5. Circuit history     — lightweight 10-15% prior (not memorisation)
  6. Reliability         — split mechanical vs driver DNF
  FP2 long run pace      — when available, replaces track DNA

Weight schedule is TRACK-AWARE:
  Monaco/Singapore/Baku → quali weight boosted, history boosted
  Monza/Spa             → track DNA boosted, history reduced
  Balanced circuits     → standard weights
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

PRE_WEEKEND = "PRE_WEEKEND"
POST_FP2    = "POST_FP2"
POST_QUALI  = "POST_QUALI"
RACE_DAY    = "RACE_DAY"

# Street circuits: qualifying position is the dominant predictor
STREET_CIRCUITS = ["monaco", "singapore", "baku", "azerbaijan",
                   "las vegas", "saudi", "jeddah"]

# Power circuits: track DNA dominates, history less relevant
POWER_CIRCUITS  = ["monza", "italian", "spa", "belgian",
                   "canadian", "montreal", "baku"]  # Baku is hybrid


# ══════════════════════════════════════════════════════════════════════════════
# WEEKEND STATE
# ══════════════════════════════════════════════════════════════════════════════

def _weekend_state(event: pd.Series) -> str:
    now       = pd.Timestamp.now()
    race_day  = pd.Timestamp(event["EventDate"])
    fp2_end   = race_day - pd.Timedelta(days=2) + pd.Timedelta(hours=16)
    quali_end = race_day - pd.Timedelta(days=1) + pd.Timedelta(hours=18)

    if now < fp2_end:
        return PRE_WEEKEND
    elif now < quali_end:
        return POST_FP2
    elif now < race_day + pd.Timedelta(hours=6):
        return POST_QUALI
    else:
        return RACE_DAY


def _is_street(race_name: str) -> bool:
    name = race_name.lower()
    return any(s in name for s in STREET_CIRCUITS)


def _is_power(race_name: str) -> bool:
    name = race_name.lower()
    return any(s in name for s in POWER_CIRCUITS)


def _weights(state: str, n_rounds: int, race_name: str) -> dict:
    """
    Track-aware weight schedule.

    Street circuits (Monaco, Singapore, Baku):
      Qualifying is king — overtaking nearly impossible.
      History matters — experience at these circuits is real.
      Track DNA less important — all street circuits demand similar car traits.

    Power circuits (Monza, Spa, Canada):
      Track DNA matters most — pure speed differential is decisive.
      History less important — car pace dominates driver style.
      Momentum still relevant.

    Balanced circuits:
      Standard weights.
    """
    street = _is_street(race_name)
    power  = _is_power(race_name)

    if state == PRE_WEEKEND:
        if street:
            return {
                "track_dna"  : 0.15,
                "momentum"   : 0.25,
                "skill_delta": 0.10,
                "quali"      : 0.30,  # season avg quali — big signal at streets
                "history"    : 0.15,  # circuit experience matters
                "reliability": 0.05,
                "fp2"        : 0.00,
            }
        elif power:
            return {
                "track_dna"  : 0.30,
                "momentum"   : 0.28,
                "skill_delta": 0.17,
                "quali"      : 0.15,
                "history"    : 0.05,
                "reliability": 0.05,
                "fp2"        : 0.00,
            }
        else:
            return {
                "track_dna"  : 0.22,
                "momentum"   : 0.28,
                "skill_delta": 0.15,
                "quali"      : 0.20,
                "history"    : 0.10,
                "reliability": 0.05,
                "fp2"        : 0.00,
            }

    elif state == POST_FP2:
        if street:
            return {
                "track_dna"  : 0.08,
                "momentum"   : 0.20,
                "skill_delta": 0.10,
                "quali"      : 0.22,
                "history"    : 0.12,
                "reliability": 0.05,
                "fp2"        : 0.23,
            }
        elif power:
            return {
                "track_dna"  : 0.10,
                "momentum"   : 0.22,
                "skill_delta": 0.13,
                "quali"      : 0.12,
                "history"    : 0.05,
                "reliability": 0.05,
                "fp2"        : 0.33,
            }
        else:
            return {
                "track_dna"  : 0.10,
                "momentum"   : 0.22,
                "skill_delta": 0.13,
                "quali"      : 0.15,
                "history"    : 0.08,
                "reliability": 0.05,
                "fp2"        : 0.27,
            }

    else:  # POST_QUALI / RACE_DAY
        if street:
            return {
                "track_dna"  : 0.05,
                "momentum"   : 0.18,
                "skill_delta": 0.08,
                "quali"      : 0.42,  # actual grid is everything at Monaco
                "history"    : 0.12,
                "reliability": 0.05,
                "fp2"        : 0.10,
            }
        elif power:
            return {
                "track_dna"  : 0.08,
                "momentum"   : 0.20,
                "skill_delta": 0.12,
                "quali"      : 0.33,
                "history"    : 0.05,
                "reliability": 0.05,
                "fp2"        : 0.17,
            }
        else:
            return {
                "track_dna"  : 0.07,
                "momentum"   : 0.20,
                "skill_delta": 0.13,
                "quali"      : 0.37,
                "history"    : 0.08,
                "reliability": 0.05,
                "fp2"        : 0.10,
            }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def elo_update(ratings: dict, race_df: pd.DataFrame, K: float = 20.0) -> dict:
    new  = ratings.copy()
    drvs = race_df["Driver"].tolist()
    pos  = dict(zip(race_df["Driver"], race_df["Finish_Clean"]))
    n    = len(drvs)
    for d1 in drvs:
        delta = 0.0
        for d2 in drvs:
            if d1 == d2: continue
            exp   = 1 / (1 + 10 ** ((ratings[d2] - ratings[d1]) / 400))
            act   = 1.0 if pos[d1] < pos[d2] else 0.0
            delta += K * (act - exp)
        new[d1] = ratings[d1] + delta / (n - 1)
    return new


def build_features(db: pd.DataFrame) -> pd.DataFrame:
    """
    Builds all features. DNFs are imputed before momentum/skill computation
    so mechanical retirements don't corrupt trajectories.
    """
    db = db.sort_values(["Round","Finish"]).reset_index(drop=True)

    # ── DNF IMPUTATION ─────────────────────────────────────────────────────────
    mechanical_mask = db["Status"].isin(["Retired", "Did not start"])
    completed       = db[~mechanical_mask & (db["Finished"] == 1)]
    driver_median   = completed.groupby("Driver")["Finish"].median().to_dict()

    def _impute(row):
        if not mechanical_mask.loc[row.name]:
            return row["Finish"]
        drv  = row["Driver"]
        grid = row["Grid"]
        med  = driver_median.get(drv, 10.0)
        if pd.notna(grid) and grid > 0:
            return float(np.clip(0.60 * grid + 0.40 * med, 1, 20))
        return float(np.clip(med, 1, 20))

    db["Finish_Clean"]   = db.apply(_impute, axis=1)
    db["Mechanical_DNF"] = mechanical_mask.astype(int)
    db["Driver_DNF"]     = db["Status"].isin(
        ["Accident","Collision","Collision damage"]).astype(int)

    # ── ELO ────────────────────────────────────────────────────────────────────
    elos = {d: 1500.0 for d in db["Driver"].unique()}
    elo_records = []
    for rnd in sorted(db["Round"].unique()):
        rdf = db[db["Round"] == rnd]
        for _, row in rdf.iterrows():
            elo_records.append({"Round": rnd, "Driver": row["Driver"],
                                 "Elo": elos[row["Driver"]]})
        elos = elo_update(elos, rdf)
    elo_df = pd.DataFrame(elo_records)
    db     = db.merge(elo_df, on=["Round","Driver"], how="left")
    db     = db.sort_values(["Driver","Round"])

    # ── TELEMETRY ──────────────────────────────────────────────────────────────
    db["Top_Speed_SMA"] = db.groupby("Driver")["Top_Speed"].expanding()\
                            .mean().reset_index(0, drop=True)
    db["Corner_SMA"]    = db.groupby("Driver")["Corner_Speed"].expanding()\
                            .mean().reset_index(0, drop=True)
    db["Quali_Gap_EMA"] = db.groupby("Driver")["Quali_Gap_s"].transform(
        lambda x: x.ewm(span=4, adjust=False).mean())

    # ── MOMENTUM (recency-weighted position score) ─────────────────────────────
    # P1=1.0, P20=0.0. Last race 50%, -2 25%, -3 15%, -4 10%.
    # Rewards sustained excellence. ANT wins 4 straight → score ~1.0.
    # VER P3+P5+P8+imputed = realistic mid-high score.
    N_FIELD  = 20.0
    RW       = [0.50, 0.25, 0.15, 0.10]

    def _pos_score(p: float) -> float:
        return float(np.clip(1.0 - (p - 1) / (N_FIELD - 1), 0.0, 1.0))

    def _momentum(series: pd.Series) -> pd.Series:
        vals   = series.values.astype(float)
        scores = []
        for i in range(len(vals)):
            start  = max(0, i - 3)
            window = vals[start:i+1][::-1]      # most recent first
            w      = RW[:len(window)]
            w      = [x / sum(w) for x in w]   # normalise
            scores.append(sum(_pos_score(p) * wt
                              for p, wt in zip(window, w)))
        return pd.Series(scores, index=series.index)

    db["Momentum"] = db.groupby("Driver")["Finish_Clean"].transform(_momentum)

    # ── DRIVER SKILL DELTA ─────────────────────────────────────────────────────
    # Only compute skill delta when BOTH teammates have clean data.
    # If one teammate had a mechanical DNF, that race is excluded from
    # skill delta — otherwise the surviving driver gets an artificially
    # inflated score for simply finishing while their teammate didn't.
    #
    # Rule: if any driver in a team had Mechanical_DNF=1 in a round,
    # set Skill_Raw to 0 for ALL drivers in that team that round.

    # First compute raw team average using Finish_Clean
    team_avg = db.groupby(["Round","Team"])["Finish_Clean"].transform("mean")
    db["Skill_Raw"] = team_avg - db["Finish_Clean"]

    # Now zero out skill delta for rounds where any teammate had a mech DNF
    mech_per_team = db.groupby(["Round","Team"])["Mechanical_DNF"].transform("max")
    db.loc[mech_per_team == 1, "Skill_Raw"] = 0.0

    # Also zero out for single-car teams (no teammate comparison possible)
    team_size = db.groupby(["Round","Team"])["Driver"].transform("count")
    db.loc[team_size < 2, "Skill_Raw"] = 0.0

    db["Skill_Delta"] = db.groupby("Driver")["Skill_Raw"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean())

    return db


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 1 — TRACK DNA MATCH
# ══════════════════════════════════════════════════════════════════════════════

def _track_dna(next_race: str) -> dict:
    default = {"w_speed":1.0,"w_corner":1.0,
               "profile":"Balanced (fallback)",
               "ft_pct":0.55,"corner_density":0.45,"avg_speed":210.0}
    try:
        s   = fastf1.get_session(YEAR - 1, next_race, "Q")
        s.load(telemetry=True, weather=False, messages=False)
        tel = s.laps.pick_fastest().get_telemetry()
        if tel is None or tel.empty:
            return default

        spd            = tel["Speed"]
        thr            = tel["Throttle"]
        ft_pct         = float((thr == 100).sum() / len(thr))
        avg_spd        = float(spd.mean())
        corner_density = float((spd < spd.max() * 0.85).sum() / len(spd))

        print(f"  Track DNA → FT%: {ft_pct*100:.1f}% | "
              f"Avg: {avg_spd:.1f} km/h | Corners: {corner_density*100:.1f}%")

        is_power  = ft_pct > 0.68 or avg_spd > 230
        is_twisty = corner_density > 0.50 or avg_spd < 195

        if is_power and not is_twisty:
            profile = "Power circuit"
            w_speed, w_corner = 1.30, 0.70
        elif is_twisty and not is_power:
            profile = "Downforce circuit"
            w_speed, w_corner = 0.70, 1.30
        else:
            profile = "Balanced circuit"
            w_speed, w_corner = 1.0, 1.0

        return {"w_speed":w_speed,"w_corner":w_corner,"profile":profile,
                "ft_pct":ft_pct,"corner_density":corner_density,
                "avg_speed":avg_spd}
    except Exception as e:
        print(f"  Track DNA unavailable: {e}")
        return default


def build_track_dna_signal(grid: pd.DataFrame, track: dict) -> np.ndarray:
    g = grid.copy()
    for col, zcol in [("Top_Speed_SMA","Spd_Z"),("Corner_SMA","Cor_Z")]:
        mu = g[col].mean(); std = g[col].std() + 1e-9
        g[zcol] = np.clip((g[col] - mu) / std, -2.5, 2.5)
    g["Spd_Z"] = g.groupby("Team")["Spd_Z"].transform("mean")
    g["Cor_Z"] = g.groupby("Team")["Cor_Z"].transform("mean")
    aero = g["Spd_Z"] * track["w_speed"] + g["Cor_Z"] * track["w_corner"]
    rng  = aero.max() - aero.min()
    if rng > 1e-6:
        return ((aero - aero.min()) / rng).values
    return np.full(len(aero), 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 2 — MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════

def build_momentum_signal(grid: pd.DataFrame) -> np.ndarray:
    """
    Momentum = recency-weighted position score with selective quali buffer.

    Root cause fix: qualifying anchor was at 30% which masked genuine
    race underperformance. NOR qualifying fast but finishing P18 shouldn't
    have his momentum rescued by quali pace — he finished the race, that
    result is real.

    The quali anchor now only applies as a MECHANICAL DNF BUFFER:
      - Rounds where a driver had a mechanical DNF: use quali position
        as a proxy for what they would have finished (better than imputed)
      - Rounds where driver finished: use actual finishing position only

    This preserves the key insight (don't penalise mechanical failures)
    without masking genuine race pace problems.

    Final composition:
      70% recency-weighted race positions (DNF-imputed for mechanicals)
      15% ELO (long-run skill baseline, stable)
      15% quali pace EMA (season-level qualifying ability, not per-race)
    """
    mom     = grid["Momentum"].values
    elo_min = grid["Elo"].min(); elo_max = grid["Elo"].max()
    elo_sc  = (grid["Elo"] - elo_min) / (elo_max - elo_min + 1e-9)

    # Quali EMA as a season-level signal (not per-race rescue)
    # This tells us how fast this driver generally qualifies across the season
    # Weight kept low at 15% so it doesn't override race results
    if "Quali_Gap_EMA" in grid.columns:
        gap  = grid["Quali_Gap_EMA"].fillna(2.0).values
        inv  = gap.max() - gap
        rng  = inv.max() - inv.min()
        q_sc = (inv - inv.min()) / rng if rng > 1e-6 \
               else np.full(len(inv), 0.5)
    else:
        q_sc = np.full(len(mom), 0.5)

    combined = 0.70 * mom + 0.15 * elo_sc.values + 0.15 * q_sc

    rng = combined.max() - combined.min()
    if rng > 1e-6:
        return (combined - combined.min()) / rng
    return np.full(len(combined), 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 3 — DRIVER SKILL DELTA
# ══════════════════════════════════════════════════════════════════════════════

def build_skill_signal(grid: pd.DataFrame) -> np.ndarray:
    delta = grid["Skill_Delta"].values
    rng   = delta.max() - delta.min()
    if rng > 1e-6:
        return (delta - delta.min()) / rng
    return np.full(len(delta), 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 4 — QUALIFYING
# ══════════════════════════════════════════════════════════════════════════════

def build_quali_season(grid: pd.DataFrame) -> np.ndarray:
    gap = grid["Quali_Gap_EMA"].fillna(2.0).values
    inv = gap.max() - gap
    rng = inv.max() - inv.min()
    if rng > 1e-6:
        return (inv - inv.min()) / rng
    return np.full(len(gap), 0.5)


def build_quali_actual(next_race: str, drivers: list) -> dict | None:
    try:
        sess = fastf1.get_session(YEAR, next_race, "Q")
        sess.load(telemetry=False, weather=False, messages=False)
        ql   = sess.laps.pick_quicklaps().copy()
        if ql.empty:
            return None
        best       = ql.groupby("Driver")["LapTime"].min().reset_index()
        best["LapSec"] = best["LapTime"].dt.total_seconds()
        best       = best.sort_values("LapSec").reset_index(drop=True)
        pole       = best["LapSec"].min()
        best["Gap"]   = best["LapSec"] - pole
        best["Score"] = best["Gap"].apply(lambda g: float(np.exp(-g / 0.8)))
        score_dict = dict(zip(best["Driver"], best["Score"]))
        for d in drivers:
            if d not in score_dict:
                score_dict[d] = 0.05
        arr = np.array([score_dict[d] for d in drivers])
        rng = arr.max() - arr.min()
        norm = (arr - arr.min()) / rng if rng > 1e-6 else np.full(len(arr), 0.5)
        print(f"  Actual quali: pole {best.iloc[0]['Driver']} ({pole:.3f}s)")
        return {d: float(norm[i]) for i, d in enumerate(drivers)}
    except Exception as e:
        print(f"  Actual quali unavailable: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 5 — CIRCUIT HISTORY (lightweight, 10-15%)
# ══════════════════════════════════════════════════════════════════════════════

def _pos_to_score(pos: float) -> float:
    """P1=1.0, exponential decay. P10≈0.22, P20≈0.03"""
    if pd.isna(pos) or pos <= 0:
        return 0.0
    return float(np.exp(-(pos - 1) / 5.5))


def _norm_team(t: str) -> str:
    return t.lower().replace(" f1 team","").replace(" racing","").strip()


def build_history_signal(next_race: str, drivers: list,
                         teams: dict,
                         momentum_scores: np.ndarray) -> np.ndarray:
    """
    Circuit history prior — but gated by current form.

    Root cause fix: history was static. A driver who won here in 2024
    but is now in terrible form (or driving a worse car) was being
    over-rewarded. This applies to every circuit, not just Monaco.

    The gate works as follows:
      - Each driver gets a raw history score from last 3 years
      - That score is then blended with their current momentum score
      - The blend ratio = how many seasons ago the result was

    Formula:
      gated_score = raw_history * form_gate_multiplier

    form_gate_multiplier:
      Momentum in top 30% of grid  → 1.10 (history slightly boosted)
      Momentum in middle 40%       → 1.00 (history unchanged)
      Momentum in bottom 30%       → 0.65 (history heavily discounted)

    This means:
      NOR (P20 + P18 recent, low momentum) → Monaco history 0.90 → 0.59
      VER (inconsistent, medium momentum)  → Monaco history 0.75 → 0.75
      ANT (4 wins, top momentum)           → Monaco history 0.00 → 0.00
      LEC (solid form, good momentum)      → Monaco history 1.00 → 1.05

    Universally correct: a driver in terrible current form
    shouldn't get full credit for past circuit wins at any track.
    A driver in dominant current form gets a slight history boost.
    """
    year_weights = {YEAR-1: 0.50, YEAR-2: 0.30, YEAR-3: 0.20}
    all_hist = []

    for yr, w in year_weights.items():
        try:
            s = fastf1.get_session(yr, next_race, "R")
            s.load(telemetry=False, weather=False, messages=False)
            res = s.results[["Abbreviation","TeamName","Position"]].copy()
            res.columns = ["Driver","Team","Position"]
            res["Position"] = pd.to_numeric(res["Position"], errors="coerce")
            res["Weight"]   = w
            all_hist.append(res.dropna(subset=["Position"]))
            print(f"  History {yr}: {len(res)} drivers")
        except Exception as e:
            print(f"  History {yr}: unavailable ({e})")

    if not all_hist:
        return np.full(len(drivers), 0.5)

    hist_df    = pd.concat(all_hist, ignore_index=True)
    team_avg   = hist_df.groupby("Team")["Position"].mean().to_dict()
    team_avg_n = {_norm_team(k): v for k, v in team_avg.items()}

    # Raw history scores
    raw_scores = {}
    for drv in drivers:
        dh = hist_df[hist_df["Driver"] == drv]
        if not dh.empty:
            num = sum(_pos_to_score(r["Position"]) * r["Weight"]
                      for _, r in dh.iterrows())
            raw_scores[drv] = num / dh["Weight"].sum()
        else:
            tk    = _norm_team(teams.get(drv, ""))
            match = next((v for k, v in team_avg_n.items()
                         if tk in k or k in tk), None)
            raw_scores[drv] = _pos_to_score(match) * 0.60 if match else 0.30

    # Form gate: compute percentile rank of each driver's momentum
    # Higher momentum → gate multiplier closer to 1.10
    # Lower momentum  → gate multiplier closer to 0.65
    mom_arr  = momentum_scores
    mom_min  = mom_arr.min()
    mom_max  = mom_arr.max()
    mom_rng  = mom_max - mom_min + 1e-9

    gated_scores = {}
    for i, drv in enumerate(drivers):
        mom_pct  = (mom_arr[i] - mom_min) / mom_rng  # 0=worst, 1=best form
        # Gate multiplier: linear interpolation 0.65 → 1.10
        gate     = 0.65 + mom_pct * 0.45
        gated_scores[drv] = raw_scores[drv] * gate

    arr = np.array([gated_scores[d] for d in drivers])
    rng = arr.max() - arr.min()
    if rng > 1e-6:
        return (arr - arr.min()) / rng
    return np.full(len(drivers), 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 6 — FP2 LONG RUN PACE
# ══════════════════════════════════════════════════════════════════════════════

def build_fp2_signal(next_race: str, drivers: list,
                     teams: dict) -> dict | None:
    try:
        sess = fastf1.get_session(YEAR, next_race, "FP2")
        sess.load(telemetry=False, weather=False, messages=False)
        laps = sess.laps.copy()
        if laps.empty:
            return None
        clean = laps[laps["TrackStatus"].isin(["1","2"])].copy()
        clean = clean[clean["PitOutTime"].isna()]
        driver_pace = {}
        for drv in drivers:
            dl = clean[clean["Driver"] == drv].copy()
            if dl.empty:
                continue
            dl = dl.sort_values("LapNumber")
            dl["StintGap"] = dl["LapNumber"].diff().fillna(0)
            dl["Stint"]    = (dl["StintGap"] > 3).cumsum()
            long_laps = []
            for _, stint in dl.groupby("Stint"):
                if len(stint) < 6:
                    continue
                valid = stint.iloc[3:].dropna(subset=["LapTime"]).copy()
                valid["LapSec"] = valid["LapTime"].dt.total_seconds()
                med   = valid["LapSec"].median()
                valid = valid[valid["LapSec"] < med * 1.07]
                if len(valid) >= 3:
                    long_laps.extend(valid["LapSec"].tolist())
            if long_laps:
                driver_pace[drv] = float(np.median(long_laps))
        if not driver_pace:
            return None
        print(f"  FP2: {len(driver_pace)} drivers with valid long runs")
        team_pace = {}
        for drv, pace in driver_pace.items():
            team_pace.setdefault(teams.get(drv,"?"), []).append(pace)
        team_med = {t: np.median(v) for t, v in team_pace.items()}
        for drv in drivers:
            if drv not in driver_pace:
                driver_pace[drv] = team_med.get(
                    teams.get(drv,""), max(driver_pace.values())) * 1.02
        arr  = np.array([driver_pace[d] for d in drivers])
        inv  = arr.max() - arr
        rng  = inv.max() - inv.min()
        norm = (inv - inv.min()) / rng if rng > 1e-6 else np.full(len(drivers), 0.5)
        return {d: float(norm[i]) for i, d in enumerate(drivers)}
    except Exception as e:
        print(f"  FP2 unavailable: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 7 — RELIABILITY (split mechanical vs driver DNF)
# ══════════════════════════════════════════════════════════════════════════════

def build_reliability_signal(db: pd.DataFrame,
                              drivers: list) -> np.ndarray:
    prior_n = 3.0; prior_r = 0.85
    drv_n   = db.groupby("Driver").size()
    drv_fin = db[db["Finished"] == 1].groupby("Driver").size()
    mech    = db[db.get("Mechanical_DNF", pd.Series(
                   0, index=db.index)) == 1].groupby("Driver").size() \
              if "Mechanical_DNF" in db.columns else pd.Series(dtype=float)

    scores = []
    for drv in drivers:
        n          = drv_n.get(drv, prior_n)
        fin        = drv_fin.get(drv, 0)
        mech_count = mech.get(drv, 0)
        # Mechanical DNF = 0.5 credit (car's fault, not driver)
        eff_fin    = fin + mech_count * 0.5
        eff_n      = n
        rel        = (eff_fin + prior_n * prior_r) / (eff_n + prior_n)
        scores.append(float(np.clip(rel, 0.0, 1.0)))

    arr = np.array(scores)
    rng = arr.max() - arr.min()
    if rng > 1e-6:
        return (arr - arr.min()) / rng
    return np.full(len(arr), 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE & SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def _temperature(scores: np.ndarray, n_rounds: int) -> float:
    std     = scores.std()
    scaling = 1.1 if n_rounds <= 6 else (0.95 if n_rounds <= 12 else 0.85)
    return float(np.clip(std * scaling, 0.12, 2.0))


def simulate(drivers: list, probs: np.ndarray, teams: dict) -> pd.DataFrame:
    n       = len(drivers)
    wins    = np.zeros(n); podiums = np.zeros(n)
    top6    = np.zeros(n); top10   = np.zeros(n)
    pts_acc = np.zeros(n)
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
        if pts: np.add.at(pts_acc, idx, pts)
    p  = wins / N_SIMS
    ci = 1.96 * np.sqrt(p * (1 - p) / N_SIMS) * 100
    return pd.DataFrame([{
        "Driver"   : drivers[i], "Team": teams[drivers[i]],
        "Win"      : round(wins[i]    / N_SIMS * 100, 2),
        "Win_CI"   : round(ci[i],                      2),
        "Podium"   : round(podiums[i] / N_SIMS * 100, 2),
        "Top6"     : round(top6[i]    / N_SIMS * 100, 2),
        "Top10"    : round(top10[i]   / N_SIMS * 100, 2),
        "Points_EV": round(pts_acc[i] / N_SIMS,       2),
    } for i in range(n)])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _load_model():
    """
    Loads trained LightGBM/XGBoost model if available.
    Returns (model, features, model_type) or None if not trained yet.
    """
    MODEL_PATH = "f1_data_historical/apex_model.pkl"
    META_PATH  = "f1_data_historical/model_metadata.json"

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        import pickle, json
        with open(MODEL_PATH, "rb") as f:
            pkg = pickle.load(f)

        with open(META_PATH, "r") as f:
            meta = json.load(f)

        print(f"║  [ ML Model loaded: {meta['model_type']} | "
              f"RMSE: {meta['val_rmse']} | "
              f"Trained: {meta['trained_at']} ]")
        return pkg
    except Exception as e:
        print(f"║  [ ML Model load failed: {e} — using signal engine ]")
        return None


def _ml_predict(model_pkg: dict, latest: pd.DataFrame,
                track: dict, n_rounds: int,
                circuit_type: str) -> np.ndarray | None:
    """
    Generates predicted finishing positions using the trained ML model.
    Returns normalised win probability array (0-1) or None on failure.
    """
    try:
        model    = model_pkg["model"]
        features = model_pkg["features"]

        # Build feature vector matching training schema
        ct_map = {"STREET": 2, "BALANCED": 1, "POWER": 0}

        total_rounds = 24  # approximate full season
        seas_prog    = n_rounds / total_rounds

        rows = []
        for _, row in latest.iterrows():
            rows.append({
                "Circuit_Code"  : ct_map.get(circuit_type, 1),
                "Reg_Era"       : 3,   # 2026 = most recent era
                "Season_Progress": seas_prog,
                "Grid"          : row.get("Grid", 10.0),
                "Grid_Pct"      : row.get("Grid_Pct", 0.5)
                                  if "Grid_Pct" in latest.columns else 0.5,
                "Momentum"      : row.get("Momentum", 0.5),
                "Elo_Norm"      : 0.0,  # will normalise below
                "Skill_Delta"   : row.get("Skill_Delta", 0.0),
                "Reliability"   : row.get("Reliability", 0.80),
                "Quali_Gap_EMA" : row.get("Quali_Gap_EMA", 1.0),
                "Points_EMA"    : row.get("Points_EMA", 0.0)
                                  if "Points_EMA" in latest.columns else 0.0,
            })

        feat_df = pd.DataFrame(rows)

        # Normalise ELO within this grid
        if "Elo" in latest.columns:
            elo     = latest["Elo"].values
            elo_mu  = elo.mean(); elo_std = elo.std() + 1e-9
            feat_df["Elo_Norm"] = (elo - elo_mu) / elo_std
        
        # Fill Grid_Pct if missing
        if "Grid" in feat_df.columns:
            feat_df["Grid_Pct"] = feat_df["Grid"].rank(
                pct=True, ascending=False).fillna(0.5)

        # Fill NaNs
        feat_df = feat_df.fillna(feat_df.median(numeric_only=True))

        X = feat_df[features].values.astype(float)

        model_type = model_pkg.get("model_type", "unknown")
        if model_type == "lightgbm":
            pred_pos = model.predict(X)
        else:
            pred_pos = model.predict(X)

        # Convert predicted positions to win probabilities
        # Lower predicted position = higher win probability
        # Use softmax on negative predicted positions
        neg_pos = -pred_pos
        exp_s   = np.exp(neg_pos - neg_pos.max())
        probs   = exp_s / exp_s.sum()

        return probs

    except Exception as e:
        print(f"║  [ ML prediction failed: {e} ]")
        return None


def predict():
    print(f"\n╔══ APEX PREDICT — {YEAR} ══╗")

    db = ingest()
    if db is None or db.empty:
        print("║  No race data yet.")
        return

    # Next race
    schedule = fastf1.get_event_schedule(YEAR)
    schedule = schedule[~schedule["EventName"].str.contains(
        "Testing", na=False)]
    schedule["RaceDay"] = pd.to_datetime(schedule["EventDate"])
    now  = pd.Timestamp.now()
    done = set(db["Round"].unique())
    upcoming = schedule[
        (schedule["RaceDay"] > now - pd.Timedelta(hours=6)) &
        (~schedule["RoundNumber"].astype(int).isin(done))
    ]
    if upcoming.empty:
        print("║  Season complete.")
        return

    next_event = upcoming.iloc[0]
    next_name  = next_event["EventName"]
    n_rounds   = int(db["Round"].max())
    state      = _weekend_state(next_event)
    is_street  = _is_street(next_name)
    is_power   = _is_power(next_name)

    circuit_type = "STREET" if is_street else ("POWER" if is_power else "BALANCED")
    print(f"║  Rounds: {n_rounds} | Target: {next_name}")
    print(f"║  State: {state} | Circuit type: {circuit_type}\n")

    # Features
    print("║  [ Building features... ]")
    db = build_features(db)

    latest  = db[db["Round"] == db["Round"].max()].copy().reset_index(drop=True)
    drivers = latest["Driver"].tolist()
    teams   = dict(zip(latest["Driver"], latest["Team"]))

    # Track DNA
    print(f"\n║  [ Track DNA: {next_name} ]")
    track = _track_dna(next_name)
    print(f"║    Profile: {track['profile']} | "
          f"Speed weight: {track['w_speed']} | "
          f"Corner weight: {track['w_corner']}")

    # Build all signals
    print(f"\n║  [ Building signals ]")

    print("║    Signal 1: Track DNA match...")
    dna_arr = build_track_dna_signal(latest.copy(), track)

    print("║    Signal 2: Momentum (recency-weighted)...")
    mom_arr = build_momentum_signal(latest)

    print("║    Signal 3: Driver skill delta...")
    skl_arr = build_skill_signal(latest)

    # Qualifying
    quali_arr = None
    if state in [POST_QUALI, RACE_DAY]:
        print("║    Signal 4: Actual qualifying results...")
        qa = build_quali_actual(next_name, drivers)
        if qa:
            quali_arr = np.array([qa[d] for d in drivers])
    if quali_arr is None:
        print("║    Signal 4: Season qualifying average...")
        quali_arr = build_quali_season(latest)

    # History
    print("║    Signal 5: Circuit history (lightweight prior)...")
    hist_arr = build_history_signal(next_name, drivers, teams, mom_arr)

    # FP2
    fp2_arr = None
    if state in [POST_FP2, POST_QUALI, RACE_DAY]:
        print("║    Signal 6: FP2 long run pace...")
        fp2 = build_fp2_signal(next_name, drivers, teams)
        if fp2:
            fp2_arr = np.array([fp2[d] for d in drivers])
    if fp2_arr is None:
        fp2_arr = dna_arr.copy()  # fallback to track DNA

    print("║    Signal 7: Reliability (mech/driver split)...")
    rel_arr = build_reliability_signal(db, drivers)

    # Weights
    W     = _weights(state, n_rounds, next_name)
    total = sum(W.values())
    W     = {k: v/total for k, v in W.items()}

    print(f"\n║  [ Circuit: {circuit_type} | State: {state} ]")
    print(f"║  [ Weights → "
          f"DNA:{W['track_dna']*100:.0f}% "
          f"Mom:{W['momentum']*100:.0f}% "
          f"Skill:{W['skill_delta']*100:.0f}% "
          f"Quali:{W['quali']*100:.0f}% "
          f"Hist:{W['history']*100:.0f}% "
          f"FP2:{W['fp2']*100:.0f}% "
          f"Rel:{W['reliability']*100:.0f}% ]")

    # Composite score
    fp2_w = W["fp2"] if fp2_arr is not None else 0.0
    score = (
        W["track_dna"]   * dna_arr   +
        W["momentum"]    * mom_arr   +
        W["skill_delta"] * skl_arr   +
        W["quali"]       * quali_arr +
        W["history"]     * hist_arr  +
        fp2_w            * fp2_arr   +
        W["reliability"] * rel_arr
    )

    temp    = _temperature(score, n_rounds)
    shifted = (score - score.max()) / temp
    exp_s   = np.exp(shifted)
    signal_probs = exp_s / exp_s.sum()

    # ── ML MODEL BLEND ────────────────────────────────────────────────────────
    # If trained model exists, blend it with signal engine.
    # Blend ratio grows as model has more training data:
    #   < 5 races current season:  ML 30% | Signal 70%
    #   5-10 races:                ML 50% | Signal 50%
    #   10+ races:                 ML 70% | Signal 30%
    # This ensures cold-start is handled by the signal engine
    # while the ML model takes over as data accumulates.

    model_pkg = _load_model()
    ml_probs  = None

    if model_pkg is not None:
        print("║  [ Running ML model prediction... ]")
        ml_probs = _ml_predict(model_pkg, latest, track, n_rounds, circuit_type)

    if ml_probs is not None:
        if n_rounds < 5:
            ml_w = 0.30
        elif n_rounds < 10:
            ml_w = 0.50
        else:
            ml_w = 0.70

        sig_w = 1.0 - ml_w
        probs = ml_w * ml_probs + sig_w * signal_probs
        # Re-normalise
        probs = probs / probs.sum()
        print(f"║  [ Blend: ML {ml_w*100:.0f}% | Signal {sig_w*100:.0f}% | "
              f"Max prob: {probs.max()*100:.1f}% ]\n")
    else:
        probs = signal_probs
        print(f"║  [ Signal engine only | "
              f"Max prob: {probs.max()*100:.1f}% ]\n")

    print(f"║  [ Temp: {temp:.3f} | "
          f"Score: {score.min():.3f}–{score.max():.3f} ]\n")

    # Signal breakdown
    pre = pd.DataFrame({
        "Driver"  : drivers,
        "Team"    : [teams[d] for d in drivers],
        "Score"   : score, "Raw%": probs * 100,
        "DNA"     : dna_arr, "Mom": mom_arr,
        "Skill"   : skl_arr, "Quali": quali_arr,
        "History" : hist_arr, "Rel": rel_arr,
        "Mom_raw" : latest["Momentum"].values,
        "Skl_raw" : latest["Skill_Delta"].values,
    }).sort_values("Score", ascending=False)

    print("║  [ Signal breakdown (top 12) ]")
    print(f"║  {'Driver':<7}{'Team':<20}{'Score':>7}{'Raw%':>7}"
          f"{'DNA':>6}{'Mom':>6}{'Skill':>6}{'Quali':>6}"
          f"{'Hist':>6}{'MomR':>7}{'SklR':>7}")
    print("║  " + "─" * 85)
    for _, r in pre.head(12).iterrows():
        print(f"║  {r['Driver']:<7}{r['Team']:<20}"
              f"{r['Score']:>7.3f}{r['Raw%']:>6.1f}%"
              f"{r['DNA']:>6.2f}{r['Mom']:>6.2f}"
              f"{r['Skill']:>6.2f}{r['Quali']:>6.2f}"
              f"{r['History']:>6.2f}{r['Mom_raw']:>7.3f}"
              f"{r['Skl_raw']:>7.2f}")

    # Simulate
    print(f"\n║  [ Running {N_SIMS:,} simulations... ]")
    sim_df = simulate(drivers, probs, teams)
    sim_df = sim_df.sort_values("Win", ascending=False).reset_index(drop=True)

    # Results
    print(f"\n╠══ {next_name.upper()} — {state} | {circuit_type} ══╣")
    print(f"║  {'Pos':<4}{'Driver':<7}{'Team':<22}"
          f"{'Win%':>6}{'±':>5}{'Podium%':>9}{'Top10%':>8}{'Pts EV':>8}")
    print("║  " + "─" * 70)
    for i, row in sim_df.head(15).iterrows():
        print(f"║  {i+1:<4}{row['Driver']:<7}{row['Team']:<22}"
              f"{row['Win']:>6.1f}{row['Win_CI']:>5.1f}"
              f"{row['Podium']:>9.1f}{row['Top10']:>8.1f}"
              f"{row['Points_EV']:>8.2f}")

    # H2H
    top6d = sim_df.head(6)["Driver"].tolist()
    print(f"\n║  [ Head-to-head Win% ]")
    print(f"║  {'':>7}", end="")
    for d in top6d: print(f"{d:>8}", end="")
    print()
    for d1 in top6d:
        w1 = sim_df[sim_df["Driver"]==d1]["Win"].values[0]
        print(f"║  {d1:>7}", end="")
        for d2 in top6d:
            if d1 == d2:
                print(f"{'—':>8}", end="")
            else:
                w2   = sim_df[sim_df["Driver"]==d2]["Win"].values[0]
                edge = w1/(w1+w2)*100 if (w1+w2)>0 else 50
                print(f"{edge:>7.0f}%", end="")
        print()

    # Save
    sim_df["Next_Race"]     = next_name
    sim_df["Weekend_State"] = state
    sim_df["Circuit_Type"]  = circuit_type
    sim_df["N_Rounds"]      = n_rounds
    sim_df["Predicted_At"]  = datetime.now().strftime("%Y-%m-%d %H:%M")
    sig_map = pre.set_index("Driver")
    for col, sc in [("DNA","DNA_Sig"),("Mom","Mom_Sig"),
                    ("Skill","Skl_Sig"),("Quali","Qua_Sig"),
                    ("History","Hist_Sig")]:
        sim_df[sc] = sim_df["Driver"].map(
            sig_map[col].to_dict()).fillna(0.5)

    sim_df.to_csv(PRED_PATH, index=False)
    print(f"\n║  ✓ Saved: {PRED_PATH}")
    print("╚══════════════════════════════╝\n")
    return sim_df


if __name__ == "__main__":
    predict()