"""
APEX PREDATOR — apex_dashboard.py
===================================
Streamlit pit wall dashboard.
Run: streamlit run apex_dashboard.py

Tabs:
  1. Live Prediction    — Win%, Podium%, Points EV with CI error bars
  2. Signal Breakdown   — per-driver contribution from each signal
  3. Physics Matrix     — Speed vs Cornering scatter
  4. Driver Form        — ELO trajectory + form EMA over season
  5. Post-Race Validator — prediction vs reality accuracy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import fastf1
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="APEX PREDATOR — F1 Strategy Engine",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

YEAR      = datetime.now().year
DATA_DIR  = f"f1_data_{YEAR}"
CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TEAM_COLORS = {
    "mercedes"      : "#00D2BE",
    "red bull"      : "#3671C6",
    "ferrari"       : "#E8002D",
    "mclaren"       : "#FF8000",
    "aston martin"  : "#358C75",
    "alpine"        : "#FF87BC",
    "williams"      : "#64C4FF",
    "racing bulls"  : "#6692FF",
    "haas"          : "#B6BABD",
    "audi"          : "#2C4A87",
    "cadillac"      : "#C9A227",
}

def tcolor(team: str) -> str:
    t = team.lower()
    for k, v in TEAM_COLORS.items():
        if k in t:
            return v
    return "#888888"


# ── loaders ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_predictions():
    p = f"{DATA_DIR}/predictions.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data(ttl=60)
def load_database():
    p = f"{DATA_DIR}/database.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data(ttl=300)
def next_race_info():
    try:
        fastf1.Cache.enable_cache(CACHE_DIR)
        s  = fastf1.get_event_schedule(YEAR)
        up = s[s["EventDate"] > pd.Timestamp.now()]
        return up.iloc[0] if not up.empty else None
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_real_results(race_name: str):
    try:
        fastf1.Cache.enable_cache(CACHE_DIR)
        s = fastf1.get_session(YEAR, race_name, "R")
        s.load(telemetry=False, weather=False, messages=False)
        r = s.results[["Abbreviation","TeamName","Position","Points"]].copy()
        r.columns = ["Driver","Team","Real_Pos","Real_Pts"]
        r["Real_Pos"] = pd.to_numeric(r["Real_Pos"], errors="coerce")
        return r.sort_values("Real_Pos").reset_index(drop=True)
    except Exception:
        return None


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏎 APEX PREDATOR")
    st.caption(f"F1 Strategy Engine · {YEAR}")
    st.divider()

    nr = next_race_info()
    if nr is not None:
        race_label = nr["EventName"]
        days_left  = max((pd.Timestamp(nr["EventDate"]) - pd.Timestamp.now()).days, 0)
        st.metric("Next Race", race_label.replace(" Grand Prix",""))
        st.metric("Days Away", days_left)
        st.metric("Round", int(nr.get("RoundNumber", 0)))
    else:
        st.info("Season complete.")

    st.divider()

    pred = load_predictions()
    db   = load_database()

    if pred is not None:
        n_r = pred["N_Rounds"].iloc[0] if "N_Rounds" in pred.columns else "?"
        st.success(f"✓ Predictions loaded ({n_r} races of data)")
        if "Predicted_At" in pred.columns:
            st.caption(f"Generated: {pred['Predicted_At'].iloc[0]}")
    else:
        st.error("No predictions. Run: python apex_predict.py")

    st.divider()

    if st.button("🔄 Re-run Prediction Engine", type="primary"):
        with st.spinner("Running full pipeline..."):
            try:
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "apex_predict.py"],
                    capture_output=True, text=True, timeout=600
                )
                st.cache_data.clear()
                if result.returncode == 0:
                    st.success("Done! Reload the page.")
                else:
                    st.error(f"Error:\n{result.stderr[-500:]}")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.caption("Manual run order:\n```\npython apex_predict.py\n```")


# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏁 Prediction",
    "📊 Signal Breakdown",
    "⚙️ Physics Matrix",
    "📈 Driver Form",
    "🎯 Post-Race Validator",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if pred is None:
        st.warning("Run `python apex_predict.py` first.")
    else:
        race_title = pred["Next_Race"].iloc[0] if "Next_Race" in pred.columns else "Next Race"
        st.header(f"🏁 {race_title}")

        n_rounds = pred["N_Rounds"].iloc[0] if "N_Rounds" in pred.columns else "?"
        st.caption(f"Based on {n_rounds} completed races · {N_SIMS:,} Monte Carlo iterations")

        colors = [tcolor(t) for t in pred["Team"]]

        # Win probability
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pred["Driver"], y=pred["Win"],
            error_y=dict(type="data",
                         array=pred["Win_CI"].tolist(),
                         visible=True,
                         color="rgba(255,255,255,0.4)",
                         thickness=1.5),
            marker_color=colors,
            text=[f"{w:.1f}%" for w in pred["Win"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Win: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title="Win Probability % (with 95% confidence intervals)",
            xaxis_title="Driver", yaxis_title="Win %",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=420, margin=dict(t=50,b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = go.Figure(go.Bar(
                x=pred["Driver"], y=pred["Podium"],
                marker_color=colors,
                text=[f"{p:.0f}%" for p in pred["Podium"]],
                textposition="outside",
            ))
            fig2.update_layout(title="Podium %", height=320,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), margin=dict(t=40,b=20))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = go.Figure(go.Bar(
                x=pred["Driver"], y=pred["Points_EV"],
                marker_color=colors,
                text=[f"{p:.1f}" for p in pred["Points_EV"]],
                textposition="outside",
            ))
            fig3.update_layout(title="Expected Points", height=320,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), margin=dict(t=40,b=20))
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Full Grid Prediction")
        disp_cols = [c for c in ["Driver","Team","Win","Win_CI","Podium","Top6","Top10","Points_EV"]
                     if c in pred.columns]
        st.dataframe(
            pred[disp_cols].style
              .background_gradient(subset=["Win"], cmap="RdYlGn")
              .format({c: "{:.1f}" for c in disp_cols
                       if c not in ["Driver","Team"]}),
            use_container_width=True, height=520,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIGNAL BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Signal Breakdown")
    st.caption("How each signal contributes to the final prediction score")

    if pred is None:
        st.warning("No predictions loaded.")
    else:
        signal_cols = [c for c in ["History","Form","Quali","Physics","Reliability"]
                       if c in pred.columns]
        if not signal_cols:
            st.info("Signal breakdown columns not in predictions CSV. "
                    "Re-run apex_predict.py to generate them.")
        else:
            top_n = pred.head(12)
            fig_sig = go.Figure()
            sig_colors = {
                "History"    : "#FF6B6B",
                "Form"       : "#4ECDC4",
                "Quali"      : "#45B7D1",
                "Physics"    : "#96CEB4",
                "Reliability": "#FFEAA7",
            }
            for sig in signal_cols:
                if sig in top_n.columns:
                    fig_sig.add_trace(go.Bar(
                        name=sig, x=top_n["Driver"],
                        y=top_n[sig],
                        marker_color=sig_colors.get(sig, "#888"),
                    ))
            fig_sig.update_layout(
                barmode="stack",
                title="Signal contribution per driver (top 12)",
                xaxis_title="Driver", yaxis_title="Normalised score",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), height=440,
            )
            st.plotly_chart(fig_sig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PHYSICS MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Aero Physics Matrix")
    db = load_database()
    if db is None:
        st.warning("No database found.")
    else:
        latest = db[db["Round"] == db["Round"].max()].copy()
        if "Top_Speed_SMA" not in latest.columns:
            # Compute on the fly
            db2 = db.sort_values(["Driver","Round"])
            db2["Top_Speed_SMA"] = db2.groupby("Driver")["Top_Speed"].expanding()\
                                      .mean().reset_index(0, drop=True)
            db2["Corner_SMA"]    = db2.groupby("Driver")["Corner_Speed"].expanding()\
                                      .mean().reset_index(0, drop=True)
            latest = db2[db2["Round"] == db2["Round"].max()].copy()

        spd_col = "Top_Speed_SMA" if "Top_Speed_SMA" in latest.columns else "Top_Speed"
        cor_col = "Corner_SMA"    if "Corner_SMA"    in latest.columns else "Corner_Speed"

        team_agg = latest.groupby("Team").agg(
            Speed    = (spd_col, "mean"),
            Cornering= (cor_col, "mean"),
        ).reset_index()

        fig_m = go.Figure()
        for _, row in team_agg.iterrows():
            fig_m.add_trace(go.Scatter(
                x=[row["Speed"]], y=[row["Cornering"]],
                mode="markers+text",
                name=row["Team"],
                text=[row["Team"].replace(" F1 Team","").replace(" Racing","")],
                textposition="top center",
                marker=dict(size=22, color=tcolor(row["Team"]),
                            line=dict(width=1.5, color="white")),
                hovertemplate=(f"<b>{row['Team']}</b><br>"
                               f"Top Speed: {row['Speed']:.1f} km/h<br>"
                               f"Corner Speed: {row['Cornering']:.1f} km/h"
                               "<extra></extra>"),
            ))
        mid_x = team_agg["Speed"].mean()
        mid_y = team_agg["Cornering"].mean()
        fig_m.add_hline(y=mid_y, line_dash="dash",
                        line_color="rgba(255,255,255,0.2)")
        fig_m.add_vline(x=mid_x, line_dash="dash",
                        line_color="rgba(255,255,255,0.2)")
        fig_m.update_layout(
            title="Team Aero DNA — Top Speed vs Corner Speed",
            xaxis_title="Avg Top Speed (km/h)",
            yaxis_title="Avg Corner Speed (km/h)",
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=520,
        )
        st.plotly_chart(fig_m, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DRIVER FORM
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Driver Form")
    db = load_database()
    if db is None:
        st.warning("No database found.")
    else:
        all_drivers = sorted(db["Driver"].unique().tolist())
        selected    = st.multiselect("Select drivers", all_drivers,
                                      default=all_drivers[:5])
        if selected:
            db2 = db.sort_values(["Driver","Round"])
            db2["Form_EMA"] = db2.groupby("Driver")["Finish"].transform(
                lambda x: x.ewm(span=5, adjust=False).mean())

            fig_f = go.Figure()
            for drv in selected:
                d    = db2[db2["Driver"] == drv]
                team = d["Team"].iloc[0]
                fig_f.add_trace(go.Scatter(
                    x=d["Round"], y=d["Form_EMA"],
                    mode="lines+markers", name=drv,
                    line=dict(color=tcolor(team), width=2.5),
                    hovertemplate=(f"<b>{drv}</b><br>Round %{{x}}<br>"
                                   f"Form EMA: %{{y:.1f}}<extra></extra>"),
                ))
            fig_f.update_layout(
                title="Driver Form EMA (lower = better finishing position)",
                xaxis_title="Round", yaxis_title="Form EMA",
                yaxis_autorange="reversed",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), height=400,
            )
            st.plotly_chart(fig_f, use_container_width=True)

        # Points accumulated
        st.subheader("Championship Points")
        pts = db.groupby("Driver")["Points"].sum().sort_values(ascending=False)
        fig_pts = go.Figure(go.Bar(
            x=pts.index, y=pts.values,
            marker_color=[tcolor(db[db["Driver"]==d]["Team"].iloc[0])
                          for d in pts.index],
            text=pts.values.astype(int),
            textposition="outside",
        ))
        fig_pts.update_layout(
            title="Season Points Tally",
            xaxis_title="Driver", yaxis_title="Points",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=380,
        )
        st.plotly_chart(fig_pts, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — POST-RACE VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Post-Race Accuracy Validator")
    db = load_database()
    if db is None:
        st.warning("No database found.")
    else:
        completed_races = db["Race"].unique().tolist()
        sel_race = st.selectbox("Select race to validate", completed_races)

        if st.button("🔄 Fetch Real Results", type="primary"):
            with st.spinner("Connecting to FIA servers..."):
                real = fetch_real_results(sel_race)

            if real is None:
                st.error("Could not fetch results.")
            elif pred is None:
                st.error("No predictions to compare against.")
            else:
                merged = pd.merge(
                    pred[["Driver","Team","Win","Podium","Points_EV"]],
                    real[["Driver","Real_Pos","Real_Pts"]],
                    on="Driver", how="inner"
                )
                merged["AI_Rank"]    = merged["Win"].rank(ascending=False).astype(int)
                merged["Pos_Error"]  = merged["AI_Rank"] - merged["Real_Pos"]
                merged               = merged.sort_values("Real_Pos").reset_index(drop=True)

                st.success(f"✓ Results loaded for {sel_race}")

                # Accuracy scatter
                fig_a = go.Figure()
                colors_a = [tcolor(t) for t in merged["Team"]]
                fig_a.add_trace(go.Scatter(
                    x=merged["AI_Rank"], y=merged["Real_Pos"],
                    mode="markers+text",
                    text=merged["Driver"],
                    textposition="top center",
                    marker=dict(size=14, color=colors_a,
                                line=dict(width=1, color="white")),
                    hovertemplate="<b>%{text}</b><br>Predicted: P%{x}<br>"
                                  "Actual: P%{y}<extra></extra>",
                ))
                mx = max(merged["Real_Pos"].max(), merged["AI_Rank"].max()) + 1
                fig_a.add_trace(go.Scatter(
                    x=[1,mx], y=[1,mx], mode="lines",
                    name="Perfect", line=dict(dash="dash",
                    color="rgba(255,255,255,0.3)"),
                ))
                fig_a.update_layout(
                    title=f"Predicted vs Actual — {sel_race}",
                    xaxis_title="AI Predicted Rank",
                    yaxis_title="Actual Position",
                    yaxis_autorange="reversed",
                    xaxis_autorange="reversed",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"), height=500,
                )
                st.plotly_chart(fig_a, use_container_width=True)

                mae         = merged["Pos_Error"].abs().mean()
                top3_correct = len(merged[
                    (merged["Real_Pos"] <= 3) & (merged["AI_Rank"] <= 3)])

                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Position Error", f"{mae:.1f} places")
                c2.metric("Podium Drivers Correct", f"{top3_correct}/3")
                c3.metric("Drivers Evaluated", len(merged))

                st.subheader("Full Comparison Table")
                disp = merged[["Driver","Team","AI_Rank","Real_Pos",
                               "Pos_Error","Win","Real_Pts"]].copy()
                disp.columns = ["Driver","Team","AI Rank","Actual",
                                "Error","AI Win%","Actual Pts"]
                st.dataframe(
                    disp.style
                      .background_gradient(subset=["Error"], cmap="RdYlGn_r")
                      .format({"AI Win%":"{:.1f}", "Error":"{:+.0f}"}),
                    use_container_width=True,
                )

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"APEX PREDATOR · {YEAR} · "
    f"FastF1 + FIA Telemetry · "
    f"Monte Carlo {N_SIMS:,} iterations · "
    f"5-signal ensemble"
)

N_SIMS = 50_000