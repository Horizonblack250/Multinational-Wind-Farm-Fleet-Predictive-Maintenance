import streamlit as st
import numpy as np
import joblib
import json
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Polaris · Turbine Health Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --card:      #1a2235;
    --border:    #1e3a5f;
    --accent:    #00d4ff;
    --green:     #00e676;
    --amber:     #ffab00;
    --orange:    #ff6d00;
    --red:       #ff1744;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem; max-width: 1400px; }

/* ── Header ── */
.polaris-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.polaris-logo {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: var(--accent);
    text-transform: uppercase;
}
.polaris-logo span { color: var(--text); opacity: 0.4; }
.polaris-sub {
    font-size: 0.75rem;
    color: var(--muted);
    font-family: var(--mono);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,214,255,0.08);
    border: 1px solid rgba(0,214,255,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 0.08em;
}
.status-dot {
    width: 7px; height: 7px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Section labels ── */
.section-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    border-left: 2px solid var(--accent);
    padding-left: 10px;
}

/* ── Risk card ── */
.risk-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    height: 100%;
}
.risk-label-text {
    font-family: var(--mono);
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 0.5rem 0;
}
.risk-confidence {
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--muted);
}

/* ── Prob bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.prob-label {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    width: 110px;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}
.prob-pct {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text);
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Sensor grid ── */
.sensor-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 1.5rem;
}
.sensor-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
}
.sensor-name {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
}
.sensor-value {
    font-family: var(--mono);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--accent);
}
.sensor-unit {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    margin-left: 3px;
}

/* ── Divider ── */
.h-rule { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] > div { padding: 0; }
.stSlider [data-baseweb="slider"] { margin-top: 0; }
label[data-testid="stWidgetLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
div[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    border-radius: 6px !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: var(--bg) !important;
    border: none !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.join(os.path.dirname(__file__), "models")
    model   = joblib.load(os.path.join(base, "polaris_risk_model.pkl"))
    encoder = joblib.load(os.path.join(base, "polaris_label_encoder.pkl"))
    with open(os.path.join(base, "polaris_features.json")) as f:
        features = json.load(f)
    return model, encoder, features

model, le, FEATURES = load_artifacts()

RISK_COLORS = {
    "Normal":      "#00e676",
    "Low Risk":    "#ffab00",
    "Medium Risk": "#ff6d00",
    "High Risk":   "#ff1744",
}
RISK_BG = {
    "Normal":      "rgba(0,230,118,0.08)",
    "Low Risk":    "rgba(255,171,0,0.08)",
    "Medium Risk": "rgba(255,109,0,0.08)",
    "High Risk":   "rgba(255,23,68,0.08)",
}
RISK_BORDER = {
    "Normal":      "rgba(0,230,118,0.3)",
    "Low Risk":    "rgba(255,171,0,0.3)",
    "Medium Risk": "rgba(255,109,0,0.3)",
    "High Risk":   "rgba(255,23,68,0.3)",
}


# ── Inference helper ────────────────────────────────────────────────────────────
def predict(wind, rotor, gen_speed, pitch, power, temp, temp_history=None):
    if temp_history is None:
        temp_history = [temp] * 15

    hist = list(temp_history) + [temp]
    feats = {
        "WindSpeed":            wind,
        "RotorSpeed":           rotor,
        "GeneratorSpeed":       gen_speed,
        "PitchDeg":             pitch,
        "PowerOutput":          power,
        "temp_roll_mean_10":    np.mean(hist[-10:]),
        "temp_roll_std_10":     np.std(hist[-10:]),
        "temp_roll_max_30":     max(hist),
        "temp_rate_of_change":  hist[-1] - hist[-2],
        "temp_lag_5":           hist[-6] if len(hist) >= 6  else hist[0],
        "temp_lag_15":          hist[-15] if len(hist) >= 15 else hist[0],
        "wind_x_temp":          wind * temp,
        "efficiency":           power / wind if wind > 0 else 0,
    }
    X = np.array([[feats[f] for f in FEATURES]])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return le.classes_[pred], proba


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="polaris-header">
    <div>
        <div class="polaris-logo">Project <span>·</span> Polaris</div>
        <div class="polaris-sub">SCADA-Driven Predictive Maintenance · Wind Turbine Health Monitor</div>
    </div>
    <div class="status-pill">
        <div class="status-dot"></div>
        MODEL ONLINE
    </div>
</div>
""", unsafe_allow_html=True)


# ── Layout ──────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Sensor Inputs</div>', unsafe_allow_html=True)
    st.caption("Enter current turbine telemetry readings to predict maintenance risk.")

    c1, c2 = st.columns(2)
    with c1:
        wind      = st.slider("Wind Speed (m/s)",        0.0, 25.0, 7.0,  0.1)
        rotor     = st.slider("Rotor Speed (rpm)",        0.0, 50.0, 20.0, 0.5)
        gen_speed = st.slider("Generator Speed (rpm)",    0.0, 2000.0, 1500.0, 10.0)
    with c2:
        pitch     = st.slider("Pitch Angle (°)",         -5.0, 90.0, 5.0,  0.5)
        power     = st.slider("Power Output (kW)",       -5.0, 500.0, 150.0, 1.0)
        temp      = st.slider("Generator Temp (°C)",     -10.0, 120.0, 35.0, 0.5)

    st.markdown('<div class="h-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Temperature History (last 15 min)</div>', unsafe_allow_html=True)
    st.caption("Optionally describe the recent temperature trend to improve prediction accuracy.")

    trend = st.selectbox(
        "Recent temperature trend",
        ["Stable (same as current)", "Slowly rising (+1°C/min)", "Rapidly rising (+3°C/min)", "Cooling down (-1°C/min)"],
        label_visibility="collapsed"
    )

    if trend == "Slowly rising (+1°C/min)":
        temp_history = [max(-10, temp - (15 - i)) for i in range(15)]
    elif trend == "Rapidly rising (+3°C/min)":
        temp_history = [max(-10, temp - 3*(15 - i)) for i in range(15)]
    elif trend == "Cooling down (-1°C/min)":
        temp_history = [min(120, temp + (15 - i)) for i in range(15)]
    else:
        temp_history = [temp] * 15

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("⚡  RUN HEALTH ASSESSMENT")


# ── Results ─────────────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Health Assessment</div>', unsafe_allow_html=True)

    if run or True:  # show default on load
        label, proba = predict(wind, rotor, gen_speed, pitch, power, temp, temp_history)
        color  = RISK_COLORS[label]
        bg     = RISK_BG[label]
        border = RISK_BORDER[label]
        conf   = proba.max() * 100

        # ── Risk badge ──
        st.markdown(f"""
        <div style="background:{bg}; border:1px solid {border}; border-radius:12px; padding:1.8rem 2rem; margin-bottom:1.2rem;">
            <div style="font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.18em;
                        text-transform:uppercase; color:{color}; opacity:0.8; margin-bottom:6px;">
                Risk Classification
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700;
                        color:{color}; letter-spacing:0.04em; line-height:1.1;">
                {label.upper()}
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:0.78rem; color:#64748b; margin-top:8px;">
                Confidence: {conf:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bars ──
        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Class Probabilities</div>', unsafe_allow_html=True)

        bar_colors = {
            "Normal":      "#00e676",
            "Low Risk":    "#ffab00",
            "Medium Risk": "#ff6d00",
            "High Risk":   "#ff1744",
        }

        prob_html = ""
        for cls, p in zip(le.classes_, proba):
            pct = p * 100
            w   = max(pct, 0.5)
            c   = bar_colors[cls]
            bold = "font-weight:700; color:#e2e8f0;" if cls == label else ""
            prob_html += f"""
            <div class="prob-row">
                <div class="prob-label" style="{bold}">{cls}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{w}%; background:{c};"></div>
                </div>
                <div class="prob-pct" style="{bold}">{pct:.1f}%</div>
            </div>"""

        st.markdown(prob_html, unsafe_allow_html=True)

        # ── Live sensor summary ──
        st.markdown('<div class="h-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Current Readings</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sensor-grid">
            <div class="sensor-tile">
                <div class="sensor-name">Wind Speed</div>
                <div class="sensor-value">{wind:.1f}<span class="sensor-unit">m/s</span></div>
            </div>
            <div class="sensor-tile">
                <div class="sensor-name">Rotor Speed</div>
                <div class="sensor-value">{rotor:.1f}<span class="sensor-unit">rpm</span></div>
            </div>
            <div class="sensor-tile">
                <div class="sensor-name">Gen Speed</div>
                <div class="sensor-value">{gen_speed:.0f}<span class="sensor-unit">rpm</span></div>
            </div>
            <div class="sensor-tile">
                <div class="sensor-name">Pitch Angle</div>
                <div class="sensor-value">{pitch:.1f}<span class="sensor-unit">°</span></div>
            </div>
            <div class="sensor-tile">
                <div class="sensor-name">Power Output</div>
                <div class="sensor-value">{power:.0f}<span class="sensor-unit">kW</span></div>
            </div>
            <div class="sensor-tile">
                <div class="sensor-name">Gen Temp</div>
                <div class="sensor-value" style="color:{'#ff1744' if temp > 68 else '#ffab00' if temp > 58 else '#00d4ff'}">
                    {temp:.1f}<span class="sensor-unit">°C</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Recommendation ──
        recs = {
            "Normal":      ("All systems nominal.", "No action required. Continue standard monitoring schedule."),
            "Low Risk":    ("Elevated temperature detected.", "Monitor generator temperature closely. Schedule inspection within 72 hours."),
            "Medium Risk": ("Abnormal operating conditions.", "Reduce load if possible. Inspect generator cooling system within 24 hours."),
            "High Risk":   ("Critical risk detected.", "Immediate inspection required. Consider taking turbine offline to prevent damage."),
        }
        title, body = recs[label]
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid {border};
                    border-left: 3px solid {color}; border-radius:0 8px 8px 0;
                    padding: 1rem 1.2rem; margin-top:0.8rem;">
            <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:{color};
                        letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">
                Recommendation
            </div>
            <div style="font-size:0.88rem; font-weight:600; color:#e2e8f0; margin-bottom:4px;">{title}</div>
            <div style="font-size:0.82rem; color:#94a3b8;">{body}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #1e3a5f; padding-top:1rem; display:flex;
            justify-content:space-between; align-items:center;">
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem;
                color:#334155; letter-spacing:0.1em;">
        PROJECT POLARIS · AVENTA AV-7 SCADA · XGBOOST CLASSIFIER · MACRO F1: 0.84
    </div>
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#334155;">
        PHASE 10 — INTERACTIVE DEMO
    </div>
</div>
""", unsafe_allow_html=True)
