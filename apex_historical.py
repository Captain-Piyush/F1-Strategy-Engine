"""
APEX PREDATOR — apex_historical.py
=====================================
One-time historical data fetcher. Pulls 2018-2025 race data from FastF1.
No telemetry — just results, qualifying times, and grid positions.
This keeps the cache under 500MB and runs in 20-30 minutes.

Run once: python apex_historical.py
Output: f1_data_historical/historical_database.csv
"""

import fastf1
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

CACHE_DIR   = "f1_cache"
OUTPUT_DIR  = "f1_data_historical"
OUTPUT_PATH = f"{OUTPUT_DIR}/historical_database.csv"

# Years to fetch. 2018 = first year FastF1 has reliable data.
YEARS = list(range(2018, 2026))

# Physical sanity cap on qualifying gap
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
    """Classify DNF as mechanical, driver, or none."""
    if pd.isna(status):
        return "none"
    s = str(status).lower()
    if any(w in s for w in ["retired", "did not start", "power", "gearbox",
                             "engine", "hydraulic", "electrical", "mechanical",
                             "brake", "suspension", "exhaust", "fuel"]):
        return "mechanical"
    if any(w in s for w in ["accident", "collision", "spin", "damage"]):
        return "driver"
    if "finished" in s or "lapped" in s or "+" in s:
        return "none"
    return "mechanical"  # default unknown retirement to mechanical


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
    street_kw = ["monaco", "singapore", "baku", "azerbaijan",
                 "las vegas", "saudi", "jeddah"]
    power_kw  = ["monza", "italian", "spa", "belgian",
                 "canadian", "montreal"]
    if any(k in name for k in street_kw):
        return "street"
    if any(k in name for k in power_kw):
        return "power"
    return "balanced"


# ── regulation era encoder ─────────────────────────────────────────────────────
# F1 has had distinct regulation eras that affect car performance relationships.
# We encode this so the model can discount older data appropriately.
def _reg_era(year: int) -> int:
    if year <= 2021:
        return 0   # hybrid V6 + old aero
    elif year <= 2021:
        return 1   # transitional
    elif year <= 2023:
        return 2   # ground effect era begins (2022 regs)
    else:
        return 3   # evolved ground effect (2024-2026 similar)


# ── main ───────────────────────────────────────────────────────────────────────

def fetch_historical():
    print("╔══ APEX HISTORICAL FETCH ══╗")
    print(f"║  Years: {YEARS[0]}–{YEARS[-1]}")
    print(f"║  Output: {OUTPUT_PATH}\n")

    # Load existing to allow resume
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        done_keys = set(
            zip(existing["Year"].astype(str),
                existing["Round"].astype(str))
        )
        print(f"║  Resuming — {len(done_keys)} race-rounds already stored\n")
    else:
        existing  = pd.DataFrame()
        done_keys = set()

    all_rows = []
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
                continue

            print(f"║    Round {rnd:>2}: {race_name}", end=" ... ", flush=True)

            try:
                # Race session — no telemetry
                sess_r = fastf1.get_session(year, event["EventName"], "R")
                sess_r.load(telemetry=False, weather=False, messages=False)

                if sess_r.results is None or sess_r.results.empty:
                    print("no results")
                    continue

                # Qualifying — no telemetry
                try:
                    sess_q = fastf1.get_session(year, event["EventName"], "Q")
                    sess_q.load(telemetry=False, weather=False, messages=False)
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
                    fin    = _finished(status)
                    dnf_t  = _dnf_type(status)
                    pos    = drv["Position"]

                    try:
                        pos = float(pos)
                    except Exception:
                        pos = np.nan

                    qgap = _quali_gap(sess_q, abbr) if has_q else np.nan
                    grid = drv["GridPosition"]
                    try:
                        grid = float(grid)
                        if grid <= 0:
                            grid = np.nan
                    except Exception:
                        grid = np.nan

                    all_rows.append({
                        "Year"         : year,
                        "Round"        : rnd,
                        "Race"         : race_name,
                        "Circuit_Type" : circ,
                        "Reg_Era"      : era,
                        "Driver"       : abbr,
                        "Team"         : drv["TeamName"],
                        "Grid"         : grid,
                        "Finish"       : pos,
                        "Status"       : status,
                        "Finished"     : fin,
                        "DNF_Type"     : dnf_t,
                        "Points"       : drv["Points"],
                        "Quali_Gap_s"  : qgap,
                    })

                total_races += 1
                print(f"✓ {len(results)} drivers")

            except Exception as e:
                failed += 1
                print(f"✗ {e}")

    # Combine with existing
    if all_rows:
        new_df   = pd.DataFrame(all_rows)
        final_df = pd.concat([existing, new_df], ignore_index=True)

        # Sanity: remove duplicate year+round+driver rows
        final_df = final_df.drop_duplicates(
            subset=["Year","Round","Driver"], keep="last")

        # Sort chronologically
        final_df = final_df.sort_values(
            ["Year","Round","Finish"]).reset_index(drop=True)

        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n║  ✓ Saved {len(final_df)} rows → {OUTPUT_PATH}")
        print(f"║  Races fetched: {total_races} | Failed: {failed}")
    else:
        print("\n║  No new data fetched.")

    print("╚══════════════════════════╝")


if __name__ == "__main__":
    fetch_historical()