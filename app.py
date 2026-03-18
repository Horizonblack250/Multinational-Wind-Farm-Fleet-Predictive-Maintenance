import streamlit as st
import numpy as np
import joblib
import json
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global-Grid · Wind Farm Health Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
    --bg:           #f4f6f9;
    --surface:      #ffffff;
    --border:       #e2e8f0;
    --border-dark:  #cbd5e1;
    --text-primary: #0f172a;
    --text-sec:     #475569;
    --text-muted:   #94a3b8;
    --accent:       #0052cc;
    --accent-light: #e8f0fe;

    --sans: 'IBM Plex Sans', sans-serif;
    --mono: 'IBM Plex Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg) !important;
    color: var(--text-primary);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1440px; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.topbar-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.topbar-sub {
    font-size: 0.78rem;
    color: var(--text-sec);
    margin-top: 3px;
    font-family: var(--mono);
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--accent-light);
    color: var(--accent);
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-family: var(--mono);
    font-weight: 600;
    letter-spacing: 0.06em;
    white-space: nowrap;
}
.live-dot {
    width: 6px; height: 6px;
    background: #16a34a;
    border-radius: 50%;
    display: inline-block;
    animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── KPI strip ── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.4rem;
}
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.kpi-label {
    font-size: 0.68rem;
    font-family: var(--mono);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1;
    font-family: var(--mono);
}
.kpi-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── Panel ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    height: 100%;
}
.panel-title {
    font-size: 0.7rem;
    font-family: var(--mono);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.7rem;
    margin-bottom: 1.2rem;
}

/* ── Result card ── */
.result-card {
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    border-width: 1px;
    border-style: solid;
}
.result-class {
    font-family: var(--mono);
    font-size: 1.9rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 4px;
}
.result-conf {
    font-family: var(--mono);
    font-size: 0.75rem;
    opacity: 0.7;
}

/* ── Prob bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
}
.prob-lbl {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-sec);
    width: 100px;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    background: var(--border);
    border-radius: 3px;
    height: 7px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 3px; }
.prob-pct {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-primary);
    width: 38px;
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
    border-radius: 6px;
    padding: 10px 12px;
}
.sensor-name {
    font-family: var(--mono);
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 3px;
}
.sensor-val {
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
}
.sensor-unit {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-left: 2px;
}

/* ── Recommendation ── */
.rec-box {
    border-radius: 0 6px 6px 0;
    border-left-width: 3px;
    border-left-style: solid;
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 0.9rem 1.1rem;
    background: var(--bg);
}
.rec-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 3px;
}
.rec-body {
    font-size: 0.76rem;
    color: var(--text-sec);
    line-height: 1.5;
}

/* ── Streamlit overrides ── */
label[data-testid="stWidgetLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    color: var(--text-sec) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.8rem !important;
    width: 100% !important;
}
.stButton > button:hover { background: #003d99 !important; }

/* ── Footer ── */
.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--mono);
    font-size: 0.65rem;
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
    "Normal":      {"color": "#0a7c59", "bg": "#ecfdf5", "bd": "#a7f3d0", "bar": "#16a34a"},
    "Low Risk":    {"color": "#b45309", "bg": "#fffbeb", "bd": "#fde68a", "bar": "#d97706"},
    "Medium Risk": {"color": "#c2410c", "bg": "#fff7ed", "bd": "#fed7aa", "bar": "#ea580c"},
    "High Risk":   {"color": "#be123c", "bg": "#fff1f2", "bd": "#fecdd3", "bar": "#e11d48"},
}

RECOMMENDATIONS = {
    "Normal":      ("All systems nominal",         "Operating within expected parameters. No action required. Continue standard monitoring interval."),
    "Low Risk":    ("Elevated temperature noted",  "Generator temperature trending above baseline. Schedule a visual inspection within 72 hours."),
    "Medium Risk": ("Abnormal operating state",    "Thermal anomaly detected. Reduce load where possible. Dispatch inspection team within 24 hours."),
    "High Risk":   ("Critical — immediate action", "Severe risk of generator failure. Take turbine offline and conduct emergency inspection now."),
}


# ── Inference ──────────────────────────────────────────────────────────────────
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


# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div>
        <div class="topbar-title">Project Global-Grid &nbsp;&middot;&nbsp; Predictive Maintenance of a Multi-National Wind Farm Fleet</div>
        <div class="topbar-sub">SCADA Telemetry Analytics &nbsp;|&nbsp; Aventa AV-7 Research Turbine &nbsp;|&nbsp; XGBoost Classifier &nbsp;|&nbsp; Macro F1: 0.84</div>
    </div>
    <div class="badge"><span class="live-dot"></span>&nbsp;MODEL ACTIVE</div>
</div>
""", unsafe_allow_html=True)


# ── KPI strip ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-strip">
    <div class="kpi-card">
        <div class="kpi-label">Weighted Accuracy</div>
        <div class="kpi-value" style="color:#0052cc;">99%</div>
        <div class="kpi-sub">Weighted F1 on test set</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Macro F1 Score</div>
        <div class="kpi-value" style="color:#0052cc;">0.84</div>
        <div class="kpi-sub">Across all 4 risk classes</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Raw Training Records</div>
        <div class="kpi-value">39M</div>
        <div class="kpi-sub">1-second SCADA telemetry</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">High Risk Prevalence</div>
        <div class="kpi-value" style="color:#be123c;">1%</div>
        <div class="kpi-sub">Realistic anomaly rate</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Main layout ────────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Sensor Input — Current Turbine Telemetry</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        wind      = st.slider("Wind Speed (m/s)",      0.0,   25.0,    7.0,  0.1)
        rotor     = st.slider("Rotor Speed (rpm)",      0.0,   50.0,   20.0,  0.5)
        gen_speed = st.slider("Generator Speed (rpm)",  0.0, 2000.0, 1500.0, 10.0)
    with c2:
        pitch     = st.slider("Pitch Angle (deg)",     -5.0,   90.0,    5.0,  0.5)
        power     = st.slider("Power Output (kW)",     -5.0,  500.0,  150.0,  1.0)
        temp      = st.slider("Generator Temp (C)",   -10.0,  120.0,   35.0,  0.5)

    st.markdown("""
    <div style="margin-top:1rem; margin-bottom:0.3rem; font-family:var(--mono);
                font-size:0.68rem; color:var(--text-muted); text-transform:uppercase;
                letter-spacing:0.1em;">
        Recent Temperature Trend (last 15 min)
    </div>""", unsafe_allow_html=True)

    trend = st.selectbox(
        "trend",
        ["Stable", "Slowly rising  (+1 C/min)", "Rapidly rising  (+3 C/min)", "Cooling down  (-1 C/min)"],
        label_visibility="collapsed"
    )

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

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Health Assessment Output</div>', unsafe_allow_html=True)

    # Risk result badge
    st.markdown(f"""
    <div class="result-card" style="background:{s['bg']}; border-color:{s['bd']};">
        <div style="font-family:var(--mono); font-size:0.65rem; text-transform:uppercase;
                    letter-spacing:0.12em; color:{s['color']}; opacity:0.8; margin-bottom:5px;">
            Risk Classification
        </div>
        <div class="result-class" style="color:{s['color']};">{label.upper()}</div>
        <div class="result-conf" style="color:{s['color']};">Confidence &nbsp;{conf:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("""
    <div style="font-family:var(--mono); font-size:0.65rem; text-transform:uppercase;
                letter-spacing:0.1em; color:var(--text-muted); margin-bottom:8px;">
        Class Probabilities
    </div>""", unsafe_allow_html=True)

    bars_html = ""
    for cls, p in zip(le.classes_, proba):
        pct  = p * 100
        w    = max(pct, 0.4)
        clr  = RISK_STYLE[cls]["bar"]
        bold = f"font-weight:600; color:{RISK_STYLE[cls]['color']};" if cls == label else ""
        bars_html += f"""
        <div class="prob-row">
            <div class="prob-lbl" style="{bold}">{cls}</div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{w}%; background:{clr};"></div>
            </div>
            <div class="prob-pct" style="{bold}">{pct:.1f}%</div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)

    # Sensor tiles
    st.markdown("""
    <div style="font-family:var(--mono); font-size:0.65rem; text-transform:uppercase;
                letter-spacing:0.1em; color:var(--text-muted); margin:1rem 0 0.5rem;">
        Live Readings
    </div>""", unsafe_allow_html=True)

    temp_color = "#be123c" if temp > 68 else "#b45309" if temp > 58 else "#0f172a"
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
        <div class="sensor-tile">
            <div class="sensor-name">Gen Temp</div>
            <div class="sensor-val" style="color:{temp_color};">
                {temp:.1f}<span class="sensor-unit">C</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation
    rec_title, rec_body = RECOMMENDATIONS[label]
    st.markdown(f"""
    <div class="rec-box" style="border-left-color:{s['color']};">
        <div class="rec-title">{rec_title}</div>
        <div class="rec-body">{rec_body}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>Project Global-Grid &nbsp;&middot;&nbsp; Aventa AV-7 SCADA &nbsp;&middot;&nbsp; 39M records resampled to 1-min &nbsp;&middot;&nbsp; Phase 10 — Interactive Demo</span>
    <span>XGBoost &nbsp;&middot;&nbsp; SMOTE &nbsp;&middot;&nbsp; Temporal Feature Engineering</span>
</div>
""", unsafe_allow_html=True)
