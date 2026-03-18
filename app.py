import streamlit as st
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="Global-Grid · Wind Farm Health Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: #f0f4f8 !important;
    color: #0d1b2a !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem !important; max-width: 1480px !important; }

/* Metric overrides */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #8896a8 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #0d1b2a !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* Slider overrides */
label[data-testid="stWidgetLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.67rem !important;
    color: #4a5568 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #0077b6, #023e8a) !important;
    color: #ffffff !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
}

/* Progress bar */
.stProgress > div > div {
    background: #0077b6 !important;
    border-radius: 4px !important;
}

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.live-dot {
    width:7px; height:7px; background:#48cae4;
    border-radius:50%; display:inline-block;
    animation: blink 1.8s ease-in-out infinite;
    vertical-align: middle;
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

RISK = {
    "Normal":      {"color": "#1b7f4f", "bg": "#e8f5ef", "bd": "#a8d5be", "bar": "#27ae60", "bar_col": "#27ae60"},
    "Low Risk":    {"color": "#b07d12", "bg": "#fef9e7", "bd": "#f5d87a", "bar": "#f0a500", "bar_col": "#f0a500"},
    "Medium Risk": {"color": "#c0420a", "bg": "#fef0e7", "bd": "#f5b98a", "bar": "#e86a2a", "bar_col": "#e86a2a"},
    "High Risk":   {"color": "#b91c3a", "bg": "#fdeef1", "bd": "#f5a0b0", "bar": "#e11d48", "bar_col": "#e11d48"},
}

RECS = {
    "Normal":      ("All systems nominal",         "Operating within expected parameters. No action required. Continue standard monitoring interval."),
    "Low Risk":    ("Elevated readings detected",  "One or more sensors trending above baseline. Schedule a visual inspection within 72 hours."),
    "Medium Risk": ("Abnormal operating state",    "Multiple anomaly signals detected. Reduce load where possible. Dispatch inspection within 24 hours."),
    "High Risk":   ("Critical — immediate action", "Severe risk of turbine failure. Take turbine offline and conduct emergency inspection now."),
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


# ── Hero (only this stays as raw HTML — it's a design element, not a component) ──
st.markdown("""
<div style="background:linear-gradient(135deg,#0077b6 0%,#023e8a 60%,#03045e 100%);
            border-radius:0 0 16px 16px; padding:1.6rem 2.2rem 1.4rem;
            margin:0 0 1.4rem; display:flex; align-items:center;
            justify-content:space-between;">
    <div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    letter-spacing:0.2em; text-transform:uppercase;
                    color:#90e0ef; margin-bottom:5px;">
            Wind Farm Predictive Analytics Platform
        </div>
        <div style="font-size:1.25rem; font-weight:700; color:#ffffff;
                    line-height:1.3; margin-bottom:5px;">
            Project Global-Grid &nbsp;&middot;&nbsp;
            Predictive Maintenance of a Multi-National Wind Farm Fleet
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
                    color:#90e0ef; opacity:0.85;">
            SCADA Telemetry &nbsp;|&nbsp; Aventa AV-7 Research Turbine
            &nbsp;|&nbsp; Phase 10 &mdash; Interactive Demo
        </div>
    </div>
    <div style="display:flex; flex-direction:column; align-items:flex-end; gap:6px;">
        <div style="background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.25);
                    border-radius:20px; padding:5px 14px; font-family:'IBM Plex Mono',monospace;
                    font-size:0.7rem; color:#ffffff; letter-spacing:0.08em;">
            <span class="live-dot"></span> &nbsp;MODEL ACTIVE
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                    color:rgba(144,224,239,0.7);">
            XGBoost &nbsp;&middot;&nbsp; SMOTE &nbsp;&middot;&nbsp; Multi-Sensor Scoring
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPI strip — native st.metric ───────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Weighted Accuracy",   "99%",  help="Weighted F1 on held-out test set")
k2.metric("Macro F1 Score",      "0.97", help="Across all 4 risk classes")
k3.metric("Raw Training Records","39M",  help="1-second SCADA telemetry")
k4.metric("High Risk F1",        "0.90", help="607 real test samples")

st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)


# ── Main layout ────────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                text-transform:uppercase; letter-spacing:0.12em; color:#4a5568;
                font-weight:600; border-bottom:2px solid #0077b6;
                padding-bottom:0.5rem; margin-bottom:1.2rem;">
        Sensor Input — Current Turbine Telemetry
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        wind      = st.slider("Wind Speed (m/s)",      0.0,   25.0,    7.0,  0.1)
        rotor     = st.slider("Rotor Speed (rpm)",      0.0,   67.0,   30.0,  0.5)
        gen_speed = st.slider("Generator Speed (rpm)",  0.0, 2000.0, 1500.0, 10.0)
    with c2:
        pitch     = st.slider("Pitch Angle (deg)",     -5.0,   90.0,    5.0,  0.5)
        power     = st.slider("Power Output (kW)",      0.0,  500.0,  150.0,  1.0)
        temp      = st.slider("Generator Temp (C)",   -10.0,  120.0,   35.0,  0.5)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.67rem;
                color:#8896a8; text-transform:uppercase; letter-spacing:0.1em;
                margin:1rem 0 0.3rem;">
        Recent Temperature Trend (last 15 min)
    </div>
    """, unsafe_allow_html=True)

    trend = st.selectbox("trend",
        ["Stable", "Slowly rising (+1 C/min)",
         "Rapidly rising (+3 C/min)", "Cooling down (-1 C/min)"],
        label_visibility="collapsed")

    if "Slowly rising" in trend:
        temp_history = [max(-10, temp - (15 - i) * 1.0) for i in range(15)]
    elif "Rapidly rising" in trend:
        temp_history = [max(-10, temp - (15 - i) * 3.0) for i in range(15)]
    elif "Cooling" in trend:
        temp_history = [min(120, temp + (15 - i) * 1.0) for i in range(15)]
    else:
        temp_history = [temp] * 15

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.63rem;
                color:#8896a8; margin:0.8rem 0 0.6rem; line-height:1.8;
                background:#f8fafc; border:1px solid #dde3ec;
                border-radius:6px; padding:0.7rem 0.9rem;">
        <strong style="color:#4a5568;">Try these to trigger different signals:</strong><br>
        Temp &gt; 59C &rarr; Medium Risk &nbsp;|&nbsp;
        Temp &gt; 68C + rapidly rising &rarr; High Risk<br>
        High wind + power &lt; 3 kW &rarr; Low / Medium Risk &nbsp;|&nbsp;
        Low power/wind ratio &rarr; Low Risk
    </div>
    """, unsafe_allow_html=True)

    st.button("Run Health Assessment")


# ── Right panel ────────────────────────────────────────────────────────────────
with right:
    label, proba = predict(wind, rotor, gen_speed, pitch, power, temp, temp_history)
    s    = RISK[label]
    conf = proba.max() * 100

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                text-transform:uppercase; letter-spacing:0.12em; color:#4a5568;
                font-weight:600; border-bottom:2px solid {bar};
                padding-bottom:0.5rem; margin-bottom:1.2rem;">
        Health Assessment Output
    </div>
    """.replace("{bar}", s["bar"]), unsafe_allow_html=True)

    # ── Risk result card ──
    st.markdown(f"""
    <div style="background:{s['bg']}; border:1px solid {s['bd']}; border-radius:10px;
                padding:1.3rem 1.5rem; margin-bottom:1.2rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                    text-transform:uppercase; letter-spacing:0.15em;
                    color:{s['color']}; opacity:0.75; margin-bottom:4px;">
            Risk Classification
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:2rem;
                    font-weight:700; color:{s['color']}; margin-bottom:3px;">
            {label.upper()}
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.73rem;
                    color:{s['color']}; opacity:0.65;">
            Confidence &nbsp;{conf:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bars — native st.progress ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                text-transform:uppercase; letter-spacing:0.12em;
                color:#8896a8; margin-bottom:0.5rem;">
        Class Probabilities
    </div>
    """, unsafe_allow_html=True)

    for cls, p in zip(le.classes_, proba):
        pct = p * 100
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                        color:{''+RISK[cls]['color']+'' if cls == label else '#4a5568'};
                        font-weight:{'600' if cls == label else '400'};
                        margin-bottom:2px;">{cls}</div>
            """, unsafe_allow_html=True)
            bar_color = RISK[cls]["bar_col"]
            st.markdown(f"""
            <div style="background:#eef2f7; border-radius:4px; height:8px;
                        overflow:hidden; margin-bottom:8px;">
                <div style="height:100%; border-radius:4px; width:{max(pct,0.4)}%;
                            background:{bar_color};"></div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                        text-align:right; padding-top:2px;
                        color:{''+RISK[cls]['color']+'' if cls == label else '#0d1b2a'};
                        font-weight:{'600' if cls == label else '400'};">
                {pct:.1f}%
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    # ── Live readings — native st.metric in 3 cols ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                text-transform:uppercase; letter-spacing:0.12em;
                color:#8896a8; margin-bottom:0.5rem;">
        Live Readings
    </div>
    """, unsafe_allow_html=True)

    eff = round(power / wind, 2) if wind > 0 else 0.0
    temp_suffix = " (HIGH)" if temp > 68 else " (WARN)" if temp > 58 else ""

    r1a, r1b, r1c = st.columns(3)
    r1a.metric("Wind Speed",    f"{wind:.1f} m/s")
    r1b.metric("Rotor Speed",   f"{rotor:.1f} rpm")
    r1c.metric("Gen Speed",     f"{gen_speed:.0f} rpm")

    r2a, r2b, r2c = st.columns(3)
    r2a.metric("Power Output",  f"{power:.0f} kW")
    r2b.metric("Efficiency",    f"{eff:.2f} kW/ms")
    r2c.metric(f"Gen Temp{temp_suffix}", f"{temp:.1f} C")

    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    # ── Recommendation ──
    rec_title, rec_body = RECS[label]
    st.markdown(f"""
    <div style="border-left:3px solid {s['color']};
                border-top:1px solid #dde3ec; border-right:1px solid #dde3ec;
                border-bottom:1px solid #dde3ec;
                border-radius:0 8px 8px 0;
                padding:0.9rem 1.1rem; background:#f8fafc;">
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.82rem;
                    font-weight:600; color:#0d1b2a; margin-bottom:3px;">
            {rec_title}
        </div>
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.76rem;
                    color:#4a5568; line-height:1.55;">
            {rec_body}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; margin-top:2rem; padding:1rem 0;
            border-top:1px solid #dde3ec; display:flex; justify-content:space-between;
            font-size:0.63rem; color:#8896a8;">
    <span>Project Global-Grid &nbsp;&middot;&nbsp; Aventa AV-7 SCADA
          &nbsp;&middot;&nbsp; 39M records &nbsp;&middot;&nbsp; Multi-Sensor Anomaly Scoring</span>
    <span>XGBoost &nbsp;&middot;&nbsp; SMOTE &nbsp;&middot;&nbsp;
          Macro F1: 0.97 &nbsp;&middot;&nbsp; Phase 10</span>
</div>
""", unsafe_allow_html=True)
