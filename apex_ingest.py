"""
APEX PREDATOR — apex_ingest.py
================================
Autonomous data ingestion pipeline.
Fetches race results, qualifying times, and telemetry for every
completed round of the current season. Incremental — only fetches
what is missing from the local database.

Columns produced:
  Race, Round, Driver, Team, Grid, Finish, Status, Finished,
  Points, Top_Speed, Corner_Speed, Quali_Gap_s
"""

import fastf1
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

YEAR      = datetime.now().year
DATA_DIR  = f"f1_data_{YEAR}"
DB_PATH   = f"{DATA_DIR}/database.csv"
CACHE_DIR = "f1_cache"


# Physical sanity caps — these are F1 laws, not arbitrary constants
# No modern F1 car has ever been >5s off pole in qualifying
# No F1 car has ever topped 400 km/h in race conditions
QUALI_GAP_MAX_S  = 5.0
TOP_SPEED_MAX_KMH = 400.0



# ── telemetry helpers ──────────────────────────────────────────────────────────

def _top_speed(lap) -> float:
    try:
        tel = lap.get_telemetry()
        return float(tel["Speed"].max()) if tel is not None and not tel.empty else np.nan
    except Exception:
        return np.nan


def _corner_speed(lap) -> float:
    """Median speed during cornering phases (below 85% of lap max)."""
    try:
        tel = lap.get_telemetry()
        if tel is None or tel.empty:
            return np.nan
        spd       = tel["Speed"]
        threshold = spd.max() * 0.85
        corners   = spd[spd < threshold]
        return float(corners.median()) if len(corners) >= 5 else np.nan
    except Exception:
        return np.nan


def _quali_gap(sess_q, abbr: str) -> float:
    """Gap in seconds to the fastest Q lap. NaN if no time set or gap is corrupt."""
    try:
        laps     = sess_q.laps.pick_driver(abbr).pick_quicklaps()
        if laps.empty:
            return np.nan
        best     = laps["LapTime"].min().total_seconds()
        pole     = sess_q.laps.pick_quicklaps()["LapTime"].min().total_seconds()
        gap      = round(best - pole, 3)
        # Sanity cap: anything above 5.0s is physically impossible in F1 quali
        # (slowest car is always within ~4s of pole). Flag as NaN.
        return gap if gap <= QUALI_GAP_MAX_S else np.nan
    except Exception:
        return np.nan


def _finished(status: str) -> int:
    if pd.isna(status):
        return 0
    return int(bool(pd.Series([status])
                    .str.contains(r"Finished|Lapped|\+\d", regex=True)
                    .iloc[0]))


# ── main ───────────────────────────────────────────────────────────────────────

def ingest():
    print(f"╔══ APEX INGEST — {YEAR} SEASON ══╗")

    # Load existing DB
    if os.path.exists(DB_PATH):
        db             = pd.read_csv(DB_PATH)
        done_rounds    = set(db["Round"].unique())
        print(f"║  Local DB: {len(done_rounds)} round(s) already stored")
    else:
        db          = pd.DataFrame()
        done_rounds = set()
        print("║  No local DB — full season build")

    # Get completed rounds
    schedule  = fastf1.get_event_schedule(YEAR)
    completed = schedule[
        (schedule["EventDate"] < pd.Timestamp.now()) &
        (~schedule["EventName"].str.contains("Testing", na=False))
    ]

    new_rows     = []
    updates_made = False

    for _, event in completed.iterrows():
        rnd  = int(event["RoundNumber"])
        name = event["EventName"].replace(" Grand Prix", "").strip()

        if rnd in done_rounds:
            continue

        print(f"║\n║  [ Round {rnd}: {name} ]")
        updates_made = True

        try:
            # Race session
            sess_r = fastf1.get_session(YEAR, event["EventName"], "R")
            sess_r.load(telemetry=True, weather=False, messages=False)

            # Qualifying session
            try:
                sess_q    = fastf1.get_session(YEAR, event["EventName"], "Q")
                sess_q.load(telemetry=False, weather=False, messages=False)
                has_quali = True
            except Exception:
                sess_q    = None
                has_quali = False

            results = sess_r.results
            laps    = sess_r.laps

            # Team telemetry pool (for DNF teammate proxy)
            team_pool: dict[str, dict] = {}
            for _, drv in results.iterrows():
                abbr = drv["Abbreviation"]
                team = drv["TeamName"]
                try:
                    fl = laps.pick_driver(abbr).pick_fastest()
                    if not fl.empty:
                        fl  = fl.iloc[0]
                        ts  = _top_speed(fl)
                        cs  = _corner_speed(fl)
                        if not np.isnan(ts):
                            team_pool.setdefault(team, {"ts": [], "cs": []})
                            team_pool[team]["ts"].append(ts)
                        if not np.isnan(cs):
                            team_pool.setdefault(team, {"ts": [], "cs": []})
                            team_pool[team]["cs"].append(cs)
                except Exception:
                    pass

            for _, drv in results.iterrows():
                abbr  = drv["Abbreviation"]
                team  = drv["TeamName"]
                fin   = _finished(str(drv["Status"]))

                # Telemetry
                ts, cs = np.nan, np.nan
                try:
                    fl = laps.pick_driver(abbr).pick_fastest()
                    if not fl.empty:
                        fl = fl.iloc[0]
                        ts = _top_speed(fl)
                        cs = _corner_speed(fl)
                except Exception:
                    pass

                # DNF fallback: teammate median
                pool = team_pool.get(team, {})
                if np.isnan(ts) and pool.get("ts"):
                    ts = float(np.median(pool["ts"]))
                if np.isnan(cs) and pool.get("cs"):
                    cs = float(np.median(pool.get("cs", [ts * 0.72])))

                new_rows.append({
                    "Race"        : name,
                    "Round"       : rnd,
                    "Driver"      : abbr,
                    "Team"        : team,
                    "Grid"        : drv["GridPosition"],
                    "Finish"      : drv["Position"],
                    "Status"      : str(drv["Status"]),
                    "Finished"    : fin,
                    "Points"      : drv["Points"],
                    "Top_Speed"   : ts,
                    "Corner_Speed": cs,
                    "Quali_Gap_s" : _get_quali_gap_safe(sess_q, abbr, has_quali),
                })

            print(f"║    ✓ {len(results)} drivers ingested")

        except Exception as e:
            print(f"║    ✗ Failed: {e}")

    # Save
    if updates_made and new_rows:
        new_df = pd.DataFrame(new_rows)

        # Grid-level 15th-percentile floor for any remaining NaNs
        for col in ["Top_Speed", "Corner_Speed"]:
            floor      = new_df.groupby("Race")[col].transform(
                lambda x: x.quantile(0.15))
            new_df[col] = new_df[col].fillna(floor)

        new_df["Quali_Gap_s"] = new_df["Quali_Gap_s"].fillna(2.0)

        db = pd.concat([db, new_df], ignore_index=True)
        db.to_csv(DB_PATH, index=False)
        print(f"║\n║  ✓ Database saved — {len(db)} total rows | {DB_PATH}")
    else:
        print("║\n║  ✓ Database already up to date")

    print("╚══════════════════════════════╝")
    return db


def _get_quali_gap_safe(sess_q, abbr, has_quali):
    if not has_quali or sess_q is None:
        return np.nan
    return _quali_gap(sess_q, abbr)


if __name__ == "__main__":
    ingest()