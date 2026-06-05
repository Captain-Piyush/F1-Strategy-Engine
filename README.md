# PITWALL — F1 Strategy Intelligence Engine

> A machine learning prediction system for Formula 1 race outcomes. Combines a LightGBM model trained on 8 years of historical race data with a 6-signal ensemble engine. Fully autonomous — one command predicts the next race, updates after every race, and improves throughout the season.

---

## What It Does

- **Predicts race winners** with win probability, podium %, top-10 %, and expected points for every driver
- **Updates automatically** after each race — run one command Monday morning, get the next race prediction
- **Adapts to the race weekend** — predictions update Friday after FP2, again Saturday after qualifying, with real data replacing proxies at each stage
- **Gets smarter every race** — more data = better ELO separation, better momentum signals, higher ML weight
- **Fully autonomous** — works from 2026 through 2027, 2028, and beyond with zero code changes

---

## Architecture

```
apex_ingest.py           →  Fetches race results + qualifying from FastF1 API
apex_historical.py       →  One-time fetch of 2018–2025 historical data  
apex_train.py            →  Trains LightGBM on historical + current season data
apex_predict.py          →  Unified prediction engine (run this after every race)
apex_dashboard.py        →  Streamlit pit wall dashboard (PITWALL UI)
apex_benchmark.py        →  Backtests predictions against real results
pitwall_benchmark.ipynb  →  Full benchmark analysis notebook with charts
```

---

## How It Works

### Two-Layer Prediction System

**Layer 1 — Signal Engine (6 signals)**

| Signal | Weight Pre-Weekend | Description |
|--------|-------------------|-------------|
| Qualifying pace | 30% | Season-average gap to pole. Switches to actual grid post-qualifying |
| Momentum | 25% | Recency-weighted position score — last race 50%, -2 25%, -3 15%, -4 10% |
| Circuit history | 15% | Last 3 years at this circuit (2025: 50%, 2024: 30%, 2023: 20%) |
| Track DNA match | 15% | Car's corner/speed telemetry profile vs circuit physical demands |
| Driver skill delta | 10% | How much driver outperforms their car vs teammate (mechanical DNFs excluded) |
| Reliability | 5% | Bayesian DNF rate — splits mechanical (car fault) vs driver fault |

**Layer 2 — LightGBM Model**

- Trained on 3,563 rows from 2018–2025 (178 races × 20 drivers)
- 11 performance features — no driver or team identity (learns patterns, not personalities)
- Validation RMSE: **1.578 positions** (excellent; random baseline is 5.73)
- Blended with signal engine: 20% ML early season → 60% ML by race 16

### Weekend State Detection

The engine automatically detects where it is in the race weekend:

```
PRE_WEEKEND  (before FP2 ends Friday)
  → Season-average qualifying pace as proxy
  → Circuit history signal active

POST_FP2     (Friday evening)
  → FP2 long-run pace replaces track DNA signal
  → Filters yellow flag laps, outlaps, tyre warm-up

POST_QUALI   (Saturday evening)
  → Actual grid positions replace season-average quali
  → Qualifying weight jumps to 42% at street circuits
  → This is the most accurate pre-race prediction

RACE_DAY     (Sunday)
  → Same as POST_QUALI
```

### Circuit Type Awareness

```
Street circuits (Monaco, Singapore, Baku, Las Vegas)
  → Qualifying weight boosted — overtaking near impossible
  → Circuit history weighted higher — experience matters

Power circuits (Monza, Spa, Canada)
  → Track DNA signal boosted — raw speed differential decisive
  → History weighted lower — car pace dominates

Balanced circuits (everything else)
  → Standard weights
```

### DNF Handling

Mechanical failures (engine, gearbox, hydraulics) are treated differently from driver errors:

- **Mechanical DNF**: imputed finish position = 60% grid + 40% season median. Does NOT penalise driver momentum or reliability score — the car failed, not the driver.
- **Driver DNF** (accident, collision): actual P20/P22 counted in momentum and reliability.

This means Russell's Canada retirement (Mercedes mechanical) does not tank his momentum score.

---

## Benchmark Results — 2026 Season

| Race | Prior Data | MAE | Winner | Podium | Spearman ρ |
|------|-----------|-----|--------|--------|------------|
| China | 1 race | 5.3 | ❌ | 2/3 | 0.455 |
| Japan | 2 races | **3.0** | ✅ | 1/3 | **0.720** |
| Miami | 3 races | 4.9 | ✅ | 1/3 | 0.492 |
| Canada | 4 races | 4.7 | ✅ | 1/3 | 0.468 |
| **Average** | | **4.45** | **75%** | **1.2/3** | **0.534** |

**Baseline comparison**

| System | MAE | Winner Accuracy |
|--------|-----|----------------|
| Random guess | 5.73 | ~5% |
| Championship standings | ~4.80 | — |
| **PITWALL (5 races)** | **4.45** | **75%** |
| PITWALL (projected 16 races) | ~2.5–3.0 | — |

PITWALL beats random baseline by **22%** with only 5 races of data. MAE improves as the season progresses — projected to reach 2.5–3.0 positions by race 16 as ELO separates, momentum stabilises, and the ML model takes on more weight.

---

## Installation

```bash
# Clone
git clone https://github.com/Captain-Piyush/F1-Strategy-Engine.git
cd F1-Strategy-Engine

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install fastf1 pandas numpy scikit-learn lightgbm streamlit plotly scipy matplotlib jupyter
```

---

## Usage

### First-time setup

```bash
# Step 1: Fetch 2018–2025 historical race data
# Takes 1–2 hours. Fully resumable if interrupted.
python apex_historical.py

# Step 2: Train the LightGBM model (~3 minutes)
python apex_train.py
```

### Every race weekend — one command

```bash
python apex_predict.py
```

This single command:
1. Detects any new completed race on FastF1 servers
2. Fetches results, qualifying times, and telemetry
3. Updates the race database
4. Rebuilds all features (ELO, momentum, skill delta, reliability)
5. Detects the next race on the calendar
6. Fetches historical circuit data
7. Detects weekend state (PRE/POST_FP2/POST_QUALI/RACE_DAY)
8. Runs 50,000 Monte Carlo simulations
9. Saves predictions to `f1_data_YYYY/predictions.csv`

### Launch the dashboard

```bash
streamlit run apex_dashboard.py
```

### Run benchmarks

```bash
python apex_benchmark.py
jupyter notebook pitwall_benchmark.ipynb
```

---

## Weekly Workflow (Autonomous)

```
Monday morning (after race):
  python apex_predict.py
  → Ingests race result, predicts next race

Friday evening (after FP2):
  python apex_predict.py
  → FP2 long-run pace included, prediction updates

Saturday evening (after qualifying):
  python apex_predict.py
  → Actual grid positions used, most accurate prediction

End of season (December):
  python apex_historical.py   → fetch completed season
  python apex_train.py        → retrain model with new data
```

---

## Key Design Decisions

**No driver or team identity as features**
The model learns "a driver who qualifies P1 consistently in a fast car tends to win" — not "Leclerc wins Monaco." This means the system works correctly for new drivers joining the grid and handles team rebrands (AlphaTauri → Racing Bulls) transparently.

**Recency-weighted momentum over simple averages**
Last race: 50%, two races ago: 25%, three: 15%, four: 10%. A driver winning 4 straight gets score ~1.0. A driver collapsing from P1 to P19 in their last two races gets penalised immediately. This captures hot streaks and cold spells that season averages miss.

**Bayesian reliability**
All drivers start with a weak prior of 80% finish rate. Rates update as evidence accumulates. A driver with 1 DNF in 2 races isn't rated as 50% reliable — the prior stabilises early estimates.

**Adaptive ML blend**
With 4 races of data, signal engine dominates (80%). With 16 races, ML dominates (60%). The transition is automatic based on `n_rounds`. This ensures cold-start robustness without sacrificing late-season accuracy.

**Mechanical DNF exclusion from skill delta**
When computing driver vs teammate comparison, any round where a teammate had a mechanical DNF is excluded entirely. Without this, VER finishing P5 while HAD retires mechanically looks like VER outperformed his car by 11 positions — noise, not signal.

---

## Project Structure

```
F1-Strategy-Engine/
├── apex_ingest.py            # Current season data pipeline
├── apex_historical.py        # One-time 2018–2025 data fetch
├── apex_train.py             # LightGBM training pipeline
├── apex_predict.py           # Unified prediction engine
├── apex_dashboard.py         # Streamlit dashboard
├── apex_benchmark.py         # Backtesting script
├── pitwall_benchmark.ipynb   # Benchmark analysis notebook
├── .gitignore
├── LICENSE
└── README.md

# Generated at runtime (gitignored — not committed):
f1_cache/                     # FastF1 telemetry cache (~500MB)
f1_data_2026/                 # Current season database + predictions
f1_data_historical/           # Historical database + trained model (.pkl)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data source | FastF1 (official FIA telemetry API) |
| ML model | LightGBM (gradient boosted trees) |
| Simulation | Monte Carlo · 50,000 iterations · Gumbel-max trick |
| Dashboard | Streamlit + Plotly |
| Backtesting | Custom walk-forward validation |
| Language | Python 3.12 |

---

## Limitations

- **Early season accuracy**: With fewer than 5 races, predictions rely heavily on circuit history and season-average qualifying pace. Accuracy improves significantly after race 6–8.
- **Chaotic races**: Safety cars, red flags, and first-lap incidents introduce randomness that no model can predict. Miami 2026 (MAE 4.9) is an example.
- **New circuits**: A brand-new circuit with no prior-year FastF1 data falls back to balanced circuit weights. Accuracy improves after the first race at that venue.
- **Regulation changes**: Major regulation shifts (2022, 2026) mean historical data from the previous era is less predictive. The model handles this via year-based sample weights, but the first season under new regs will always be noisier.

---

## License

MIT License — see [LICENSE](LICENSE)

---

*PITWALL · Built with FastF1, LightGBM, and 50,000 Monte Carlo simulations per race.*