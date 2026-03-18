import streamlit as st
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="Global-Grid · Wind Farm Health Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
    --bg:        #f0f4f8;
    --surface:   #ffffff;
    --border:    #dde3ec;
    --text:      #0d1b2a;
    --text-sec:  #4a5568;
    --text-muted:#8896a8;
    --sans: 'IBM Plex Sans', sans-serif;
    --mono: 'IBM Plex Mono', monospace;

    --wind-blue:   #0077b6;
    --wind-teal:   #00b4d8;
    --wind-sky:    #e8f4fd;
    --wind-green:  #2d6a4f;
    --wind-grass:  #e8f5e9;

    --normal-c:  #1b7f4f;  --normal-bg:  #e8f5ef;  --normal-bd: #a8d5be;
    --low-c:     #b07d12;  --low-bg:     #fef9e7;  --low-bd:    #f5d87a;
    --med-c:     #c0420a;  --med-bg:     #fef0e7;  --med-bd:    #f5b98a;
    --high-c:    #b91c3a;  --high-bg:    #fdeef1;  --high-bd:   #f5a0b0;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem; max-width: 1480px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0077b6 0%, #023e8a 60%, #03045e 100%);
    border-radius: 0 0 16px 16px;
    padding: 1.6rem 2.2rem 1.4rem;
    margin: 0 0 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    right: -40px; top: -60px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: rgba(0,180,216,0.12);
}
.hero::after {
    content: '';
    position: absolute;
    right: 120px; bottom: -80px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(0,119,182,0.15);
}
.hero-left { position: relative; z-index: 2; }
.hero-eyebrow {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #90e0ef;
    margin-bottom: 5px;
}
.hero-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.01em;
    line-height: 1.25;
    margin-bottom: 5px;
}
.hero-sub {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: #90e0ef;
    opacity: 0.85;
}
.hero-right {
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 5px 14px;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: #ffffff;
    letter-spacing: 0.08em;
    backdrop-filter: blur(4px);
}
.live-dot {
    width: 7px; height: 7px;
    background: #48cae4;
    border-radius: 50%;
    animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.hero-tech {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: rgba(144,224,239,0.7);
    letter-spacing: 0.06em;
}

/* ── KPI strip ── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.4rem;
}
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border-top: 3px solid transparent;
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.kpi-card-blue  { border-top-color: #0077b6; }
.kpi-card-teal  { border-top-color: #00b4d8; }
.kpi-card-green { border-top-color: #2d6a4f; }
.kpi-card-red   { border-top-color: #b91c3a; }
.kpi-icon {
    font-size: 1.3rem;
    margin-bottom: 6px;
    display: block;
}
.kpi-label {
    font-family: var(--mono);
    font-size: 0.63rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.kpi-val {
    font-family: var(--mono);
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
}

/* ── Panels ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.75rem;
    margin-bottom: 1.2rem;
}
.panel-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}
.panel-title {
    font-family: var(--mono);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-sec);
    font-weight: 600;
}

/* ── Result card ── */
.result-card {
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.2rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-card::after {
    content: '';
    position: absolute;
    right: -20px; top: -20px;
    width: 90px; height: 90px;
    border-radius: 50%;
    opacity: 0.15;
    background: currentColor;
}
.result-eyebrow {
    font-family: var(--mono);
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    opacity: 0.7;
    margin-bottom: 4px;
}
.result-label {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 3px;
}
.result-conf {
    font-family: var(--mono);
    font-size: 0.73rem;
    opacity: 0.65;
}

/* ── Prob bars ── */
.prob-section { margin-bottom: 1rem; }
.prob-section-title {
    font-family: var(--mono);
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 6px;
}
.prob-lbl {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-sec);
    width: 95px;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    background: #eef2f7;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 4px; }
.prob-pct {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text);
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Sensor tiles ── */
.sensor-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 1rem;
}
.sensor-tile {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    border-left: 3px solid #00b4d8;
}
.sensor-name {
    font-family: var(--mono);
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 3px;
}
.sensor-val {
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
}
.sensor-unit {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-muted);
    margin-left: 2px;
}

/* ── Recommendation ── */
.rec-box {
    border-radius: 0 8px 8px 0;
    border-left: 3px solid;
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 0.9rem 1.1rem;
    background: var(--bg);
}
.rec-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 3px;
}
.rec-body { font-size: 0.76rem; color: var(--text-sec); line-height: 1.55; }

/* ── Streamlit overrides ── */
label[data-testid="stWidgetLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.67rem !important;
    color: var(--text-sec) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0077b6, #023e8a) !important;
    color: #ffffff !important;
    border: none !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

/* ── Footer ── */
.footer {
    margin-top: 2rem;
    padding: 1rem 0;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 0.63rem;
    color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base    = os.path.join(os.path.dirname(__file__), "models")
    model   = joblib.load(os.path.join(base, "polaris_risk_model.pkl"))
    encoder = joblib.load(os.path.join(base, "polaris_label_encoder.pkl"))
    with open(os.path.join(base, "polaris_features.json")) as f:
        features = json.load(f)
    return model, encoder, features

model, le, FEATURES = load_artifacts()

RISK_STYLE = {
    "Normal":      {"color": "#1b7f4f", "bg": "#e8f5ef", "bd": "#a8d5be", "bar": "#27ae60"},
    "Low Risk":    {"color": "#b07d12", "bg": "#fef9e7", "bd": "#f5d87a", "bar": "#f0a500"},
    "Medium Risk": {"color": "#c0420a", "bg": "#fef0e7", "bd": "#f5b98a", "bar": "#e86a2a"},
    "High Risk":   {"color": "#b91c3a", "bg": "#fdeef1", "bd": "#f5a0b0", "bar": "#e11d48"},
}
PANEL_DOTS = {
    "Normal": "#27ae60", "Low Risk": "#f0a500",
    "Medium Risk": "#e86a2a", "High Risk": "#e11d48"
}
RECS = {
    "Normal":      ("All systems nominal",         "Operating within expected parameters. No action required. Continue standard monitoring interval."),
    "Low Risk":    ("Elevated temperature noted",  "Generator temperature trending above baseline. Schedule a visual inspection within 72 hours."),
    "Medium Risk": ("Abnormal operating state",    "Thermal anomaly detected. Reduce load where possible. Dispatch inspection team within 24 hours."),
    "High Risk":   ("Critical — immediate action", "Severe risk of generator failure. Take turbine offline and conduct emergency inspection now."),
}


def predict(wind, rotor, gen_speed, pitch, power, temp, temp_history):
    hist = list(temp_history) + [temp]
    feats = {
        "WindSpeed":           wind,
        "RotorSpeed":          rotor,
        "GeneratorSpeed":      gen_speed,
        "PitchDeg":            pitch,
        "PowerOutput":         power,
        "temp_roll_mean_10":   np.mean(hist[-10:]),
        "temp_roll_std_10":    np.std(hist[-10:]),
        "temp_roll_max_30":    max(hist),
        "temp_rate_of_change": hist[-1] - hist[-2],
        "temp_lag_5":          hist[-6]  if len(hist) >= 6  else hist[0],
        "temp_lag_15":         hist[-15] if len(hist) >= 15 else hist[0],
        "wind_x_temp":         wind * temp,
        "efficiency":          power / wind if wind > 0 else 0,
    }
    X     = np.array([[feats[f] for f in FEATURES]])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return le.classes_[pred], proba


# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <div class="hero-eyebrow">Wind Farm Predictive Analytics Platform</div>
        <div class="hero-title">Project Global-Grid<br>Predictive Maintenance of a Multi-National Wind Farm Fleet</div>
        <div class="hero-sub">SCADA Telemetry &nbsp;|&nbsp; Aventa AV-7 Research Turbine &nbsp;|&nbsp; Phase 10 — Interactive Demo</div>
    </div>
    <div class="hero-right">
        <div class="hero-badge"><span class="live-dot"></span>&nbsp;MODEL ACTIVE</div>
        <div class="hero-tech">XGBoost &nbsp;·&nbsp; SMOTE &nbsp;·&nbsp; Temporal Features</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPI strip ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-strip">
    <div class="kpi-card kpi-card-blue">
        <div class="kpi-label">Weighted Accuracy</div>
        <div class="kpi-val" style="color:#0077b6;">99%</div>
        <div class="kpi-sub">Weighted F1 on held-out test set</div>
    </div>
    <div class="kpi-card kpi-card-teal">
        <div class="kpi-label">Macro F1 Score</div>
        <div class="kpi-val" style="color:#00b4d8;">0.84</div>
        <div class="kpi-sub">Across all 4 risk classes</div>
    </div>
    <div class="kpi-card kpi-card-green">
        <div class="kpi-label">Raw Training Records</div>
        <div class="kpi-val" style="color:#2d6a4f;">39M</div>
        <div class="kpi-sub">1-second SCADA telemetry</div>
    </div>
    <div class="kpi-card kpi-card-red">
        <div class="kpi-label">High Risk Prevalence</div>
        <div class="kpi-val" style="color:#b91c3a;">1%</div>
        <div class="kpi-sub">Realistic field anomaly rate</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Main layout ────────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <div class="panel-dot" style="background:#0077b6;"></div>
            <div class="panel-title">Sensor Input — Current Turbine Telemetry</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="background:#fff; border:1px solid #dde3ec; border-radius:10px; padding:1.4rem 1.6rem; margin-top:-1rem;">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            wind      = st.slider("Wind Speed (m/s)",      0.0,   25.0,    7.0,  0.1)
            rotor     = st.slider("Rotor Speed (rpm)",      0.0,   50.0,   20.0,  0.5)
            gen_speed = st.slider("Generator Speed (rpm)",  0.0, 2000.0, 1500.0, 10.0)
        with c2:
            pitch     = st.slider("Pitch Angle (deg)",     -5.0,   90.0,    5.0,  0.5)
            power     = st.slider("Power Output (kW)",     -5.0,  500.0,  150.0,  1.0)
            temp      = st.slider("Generator Temp (C)",   -10.0,  120.0,   35.0,  0.5)

        st.markdown("""<div style="font-family:'IBM Plex Mono',monospace; font-size:0.67rem;
            color:#8896a8; text-transform:uppercase; letter-spacing:0.1em;
            margin:1rem 0 0.3rem;">Recent Temperature Trend (last 15 min)</div>""",
            unsafe_allow_html=True)

        trend = st.selectbox("trend",
            ["Stable", "Slowly rising (+1 C/min)", "Rapidly rising (+3 C/min)", "Cooling down (-1 C/min)"],
            label_visibility="collapsed")

        if "Slowly rising" in trend:
            temp_history = [max(-10, temp - (15 - i) * 1.0) for i in range(15)]
        elif "Rapidly rising" in trend:
            temp_history = [max(-10, temp - (15 - i) * 3.0) for i in range(15)]
        elif "Cooling" in trend:
            temp_history = [min(120, temp + (15 - i) * 1.0) for i in range(15)]
        else:
            temp_history = [temp] * 15

        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Run Health Assessment")
        st.markdown('</div>', unsafe_allow_html=True)


with right:
    label, proba = predict(wind, rotor, gen_speed, pitch, power, temp, temp_history)
    s    = RISK_STYLE[label]
    conf = proba.max() * 100

    # Result card
    st.markdown(f"""
    <div class="panel" style="margin-bottom:0;">
        <div class="panel-header">
            <div class="panel-dot" style="background:{PANEL_DOTS[label]};"></div>
            <div class="panel-title">Health Assessment Output</div>
        </div>

        <div class="result-card" style="background:{s['bg']}; border-color:{s['bd']}; color:{s['color']};">
            <div class="result-eyebrow">Risk Classification</div>
            <div class="result-label">{label.upper()}</div>
            <div class="result-conf">Confidence &nbsp;{conf:.1f}%</div>
        </div>

        <div class="prob-section">
            <div class="prob-section-title">Class Probabilities</div>
    """, unsafe_allow_html=True)

    bars = ""
    for cls, p in zip(le.classes_, proba):
        pct  = p * 100
        w    = max(pct, 0.4)
        clr  = RISK_STYLE[cls]["bar"]
        bold = f"font-weight:600; color:{RISK_STYLE[cls]['color']};" if cls == label else ""
        bars += f"""
        <div class="prob-row">
            <div class="prob-lbl" style="{bold}">{cls}</div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{w}%; background:{clr};"></div>
            </div>
            <div class="prob-pct" style="{bold}">{pct:.1f}%</div>
        </div>"""

    st.markdown(bars + "</div>", unsafe_allow_html=True)

    # Sensor tiles
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
        text-transform:uppercase; letter-spacing:0.12em; color:#8896a8;
        margin:0.8rem 0 0.5rem;">Live Readings</div>""", unsafe_allow_html=True)

    temp_color = "#b91c3a" if temp > 68 else "#b07d12" if temp > 58 else "#0d1b2a"
    temp_bd    = "#e11d48" if temp > 68 else "#f0a500" if temp > 58 else "#00b4d8"

    st.markdown(f"""
    <div class="sensor-grid">
        <div class="sensor-tile">
            <div class="sensor-name">Wind Speed</div>
            <div class="sensor-val">{wind:.1f}<span class="sensor-unit">m/s</span></div>
        </div>
        <div class="sensor-tile">
            <div class="sensor-name">Rotor Speed</div>
            <div class="sensor-val">{rotor:.1f}<span class="sensor-unit">rpm</span></div>
        </div>
        <div class="sensor-tile">
            <div class="sensor-name">Gen Speed</div>
            <div class="sensor-val">{gen_speed:.0f}<span class="sensor-unit">rpm</span></div>
        </div>
        <div class="sensor-tile">
            <div class="sensor-name">Pitch Angle</div>
            <div class="sensor-val">{pitch:.1f}<span class="sensor-unit">deg</span></div>
        </div>
        <div class="sensor-tile">
            <div class="sensor-name">Power Output</div>
            <div class="sensor-val">{power:.0f}<span class="sensor-unit">kW</span></div>
        </div>
        <div class="sensor-tile" style="border-left-color:{temp_bd};">
            <div class="sensor-name">Gen Temp</div>
            <div class="sensor-val" style="color:{temp_color};">
                {temp:.1f}<span class="sensor-unit">C</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation
    rec_title, rec_body = RECS[label]
    st.markdown(f"""
    <div class="rec-box" style="border-left-color:{s['color']};">
        <div class="rec-title">{rec_title}</div>
        <div class="rec-body">{rec_body}</div>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>Project Global-Grid &nbsp;&middot;&nbsp; Aventa AV-7 SCADA Dataset &nbsp;&middot;&nbsp; 39M records &nbsp;&middot;&nbsp; Resampled to 1-min averages</span>
    <span>XGBoost &nbsp;&middot;&nbsp; SMOTE &nbsp;&middot;&nbsp; Temporal Feature Engineering &nbsp;&middot;&nbsp; Phase 10</span>
</div>
""", unsafe_allow_html=True)
