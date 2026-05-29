"""
APEX PREDATOR — apex_historical.py
=====================================
One-time historical data fetcher. Pulls 2018-2025 race data from FastF1.
No telemetry — just results and qualifying times.

Handles FastF1's 500 calls/hour rate limit automatically:
  - Catches RateLimitExceededError
  - Waits 65 seconds then retries
  - Up to 5 retries per race

Run once: python apex_historical.py
Output: f1_data_historical/historical_database.csv
Resumable: already-fetched races are skipped automatically.
"""

import fastf1
import pandas as pd
import numpy as np
import os
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

CACHE_DIR   = "f1_cache"
OUTPUT_DIR  = "f1_data_historical"
OUTPUT_PATH = f"{OUTPUT_DIR}/historical_database.csv"
YEARS       = list(range(2018, 2026))

# Physical sanity cap
QUALI_GAP_MAX_S = 5.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# ── helpers ────────────────────────────────────────────────────────────────────

def _finished(status: str) -> int:
    if pd.isna(status):
        return 0
    return int(bool(pd.Series([status])
                    .str.contains(r"Finished|Lapped|\+\d", regex=True)
                    .iloc[0]))


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


def _quali_gap(sess_q, abbr: str) -> float:
    try:
        laps = sess_q.laps.pick_driver(abbr).pick_quicklaps()
        if laps.empty:
            return np.nan
        best = laps["LapTime"].min().total_seconds()
        pole = sess_q.laps.pick_quicklaps()["LapTime"].min().total_seconds()
        gap  = round(best - pole, 3)
        return gap if gap <= QUALI_GAP_MAX_S else np.nan
    except Exception:
        return np.nan


def _circuit_type(event_name: str) -> str:
    name = event_name.lower()
    if any(k in name for k in ["monaco","singapore","baku","azerbaijan",
                                "las vegas","saudi","jeddah"]):
        return "street"
    if any(k in name for k in ["monza","italian","spa","belgian",
                                "canadian","montreal"]):
        return "power"
    return "balanced"


def _reg_era(year: int) -> int:
    if year <= 2021:
        return 0
    elif year <= 2023:
        return 2
    else:
        return 3


def _load_session_with_retry(year, event_name, session_type,
                              max_retries=5, wait_s=70):
    """
    Loads a FastF1 session with automatic rate limit recovery.
    Waits wait_s seconds when rate limit is hit, then retries.
    """
    for attempt in range(max_retries):
        try:
            sess = fastf1.get_session(year, event_name, session_type)
            sess.load(telemetry=False, weather=False, messages=False)
            return sess
        except fastf1.RateLimitExceededError:
            if attempt < max_retries - 1:
                print(f"\n║    ⚠ Rate limit hit — waiting {wait_s}s "
                      f"(attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(wait_s)
            else:
                raise
        except Exception as e:
            raise e


# ── main ───────────────────────────────────────────────────────────────────────

def fetch_historical():
    print("╔══ APEX HISTORICAL FETCH ══╗")
    print(f"║  Years: {YEARS[0]}–{YEARS[-1]}")
    print(f"║  Output: {OUTPUT_PATH}\n")

    # Load existing to allow resume
    if os.path.exists(OUTPUT_PATH):
        existing  = pd.read_csv(OUTPUT_PATH)
        done_keys = set(zip(existing["Year"].astype(str),
                            existing["Round"].astype(str)))
        print(f"║  Resuming — {len(done_keys)} race-rounds already stored\n")
    else:
        existing  = pd.DataFrame()
        done_keys = set()

    all_rows    = []
    total_races = 0
    failed      = 0

    for year in YEARS:
        print(f"║  ── {year} ──")
        try:
            schedule = fastf1.get_event_schedule(year)
            schedule = schedule[
                ~schedule["EventName"].str.contains("Testing", na=False)
            ].reset_index(drop=True)
        except Exception as e:
            print(f"║    Schedule failed: {e}")
            continue

        for _, event in schedule.iterrows():
            rnd       = int(event["RoundNumber"])
            race_name = event["EventName"].replace(" Grand Prix","").strip()
            key       = (str(year), str(rnd))

            if key in done_keys:
                print(f"║    Round {rnd:>2}: {race_name} ... already stored")
                continue

            print(f"║    Round {rnd:>2}: {race_name}", end=" ... ", flush=True)

            try:
                # Race session with rate limit retry
                sess_r = _load_session_with_retry(
                    year, event["EventName"], "R")

                if sess_r.results is None or sess_r.results.empty:
                    print("no results")
                    continue

                # Qualifying session with rate limit retry
                try:
                    sess_q = _load_session_with_retry(
                        year, event["EventName"], "Q")
                    has_q  = True
                except Exception:
                    sess_q = None
                    has_q  = False

                results = sess_r.results
                circ    = _circuit_type(event["EventName"])
                era     = _reg_era(year)

                for _, drv in results.iterrows():
                    abbr   = drv["Abbreviation"]
                    status = str(drv["Status"])
                    pos    = drv["Position"]
                    grid   = drv["GridPosition"]

                    try:
                        pos = float(pos)
                    except Exception:
                        pos = np.nan

                    try:
                        grid = float(grid)
                        if grid <= 0:
                            grid = np.nan
                    except Exception:
                        grid = np.nan

                    all_rows.append({
                        "Year"        : year,
                        "Round"       : rnd,
                        "Race"        : race_name,
                        "Circuit_Type": circ,
                        "Reg_Era"     : era,
                        "Driver"      : abbr,
                        "Team"        : drv["TeamName"],
                        "Grid"        : grid,
                        "Finish"      : pos,
                        "Status"      : status,
                        "Finished"    : _finished(status),
                        "DNF_Type"    : _dnf_type(status),
                        "Points"      : drv["Points"],
                        "Quali_Gap_s" : _quali_gap(sess_q, abbr) if has_q
                                        else np.nan,
                    })

                total_races += 1
                print(f"✓ {len(results)} drivers")

                # Save incrementally every 5 races so crashes don't lose data
                if total_races % 5 == 0 and all_rows:
                    _save(existing, all_rows)
                    print(f"║    [Auto-saved at {total_races} new races]")

            except fastf1.RateLimitExceededError:
                failed += 1
                print(f"✗ rate limit exhausted after retries — skipping")
            except Exception as e:
                failed += 1
                print(f"✗ {e}")

    # Final save
    if all_rows:
        _save(existing, all_rows)

    print(f"\n║  Races fetched: {total_races} | Failed: {failed}")
    print("╚══════════════════════════╝")


def _save(existing: pd.DataFrame, new_rows: list):
    new_df   = pd.DataFrame(new_rows)
    final_df = pd.concat([existing, new_df], ignore_index=True)
    final_df = final_df.drop_duplicates(
        subset=["Year","Round","Driver"], keep="last")
    final_df = final_df.sort_values(
        ["Year","Round","Finish"]).reset_index(drop=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"║  ✓ Saved {len(final_df)} rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    fetch_historical()