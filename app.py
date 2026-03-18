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

# ── Global styles (layout + overrides only, no component classes) ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: #f0f4f8 !important;
    color: #0d1b2a !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem !important; max-width: 1480px !important; }
label[data-testid="stWidgetLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.67rem !important;
    color: #4a5568 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
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
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.live-dot {
    width: 7px; height: 7px; background: #48cae4;
    border-radius: 50%; display: inline-block;
    animation: blink 1.8s ease-in-out infinite;
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
    "Normal":      {"color": "#1b7f4f", "bg": "#e8f5ef", "bd": "#a8d5be", "bar": "#27ae60"},
    "Low Risk":    {"color": "#b07d12", "bg": "#fef9e7", "bd": "#f5d87a", "bar": "#f0a500"},
    "Medium Risk": {"color": "#c0420a", "bg": "#fef0e7", "bd": "#f5b98a", "bar": "#e86a2a"},
    "High Risk":   {"color": "#b91c3a", "bg": "#fdeef1", "bd": "#f5a0b0", "bar": "#e11d48"},
}
RECS = {
    "Normal":      ("All systems nominal",         "Operating within expected parameters. No action required. Continue standard monitoring interval."),
    "Low Risk":    ("Elevated readings detected",  "One or more sensors trending above baseline. Schedule a visual inspection within 72 hours."),
    "Medium Risk": ("Abnormal operating state",    "Multiple anomaly signals detected. Reduce load where possible. Dispatch inspection team within 24 hours."),
    "High Risk":   ("Critical — immediate action", "Severe risk of turbine failure. Take offline and conduct emergency inspection now."),
}

MONO = "font-family:'IBM Plex Mono',monospace;"
SANS = "font-family:'IBM Plex Sans',sans-serif;"


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


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0077b6 0%,#023e8a 60%,#03045e 100%);
            border-radius:0 0 16px 16px; padding:1.6rem 2.2rem 1.4rem;
            margin:0 0 1.4rem; display:flex; align-items:center;
            justify-content:space-between; position:relative; overflow:hidden;">
    <div style="position:relative; z-index:2;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    letter-spacing:0.2em; text-transform:uppercase; color:#90e0ef; margin-bottom:5px;">
            Wind Farm Predictive Analytics Platform
        </div>
        <div style="font-size:1.25rem; font-weight:700; color:#ffffff;
                    letter-spacing:-0.01em; line-height:1.25; margin-bottom:5px;">
            Project Global-Grid<br>
            Predictive Maintenance of a Multi-National Wind Farm Fleet
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#90e0ef; opacity:0.85;">
            SCADA Telemetry &nbsp;|&nbsp; Aventa AV-7 Research Turbine &nbsp;|&nbsp; Phase 10 &mdash; Interactive Demo
        </div>
    </div>
    <div style="position:relative; z-index:2; display:flex; flex-direction:column;
                align-items:flex-end; gap:6px;">
        <div style="display:inline-flex; align-items:center; gap:6px;
                    background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.25);
                    border-radius:20px; padding:5px 14px; font-family:'IBM Plex Mono',monospace;
                    font-size:0.7rem; color:#ffffff; letter-spacing:0.08em;">
            <span class="live-dot"></span>&nbsp;MODEL ACTIVE
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                    color:rgba(144,224,239,0.7); letter-spacing:0.06em;">
            XGBoost &nbsp;&middot;&nbsp; SMOTE &nbsp;&middot;&nbsp; Multi-Sensor Scoring
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPI strip ──────────────────────────────────────────────────────────────────
def kpi(top_color, val_color, label, value, sub):
    return f"""
    <div style="background:#ffffff; border:1px solid #dde3ec; border-radius:10px;
                padding:1.1rem 1.3rem; box-shadow:0 1px 3px rgba(0,0,0,0.05);
                border-top:3px solid {top_color};">
        <div style="{MONO} font-size:0.63rem; text-transform:uppercase;
                    letter-spacing:0.12em; color:#8896a8; margin-bottom:4px;">{label}</div>
        <div style="{MONO} font-size:1.75rem; font-weight:700; line-height:1;
                    margin-bottom:4px; color:{val_color};">{value}</div>
        <div style="{SANS} font-size:0.72rem; color:#8896a8;">{sub}</div>
    </div>"""

st.markdown(f"""
<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:1.4rem;">
    {kpi('#0077b6','#0077b6','Weighted Accuracy','99%','Weighted F1 on held-out test set')}
    {kpi('#00b4d8','#00b4d8','Macro F1 Score','0.97','Across all 4 risk classes')}
    {kpi('#2d6a4f','#2d6a4f','Raw Training Records','39M','1-second SCADA telemetry')}
    {kpi('#b91c3a','#b91c3a','High Risk F1','0.90','607 real test samples')}
</div>
""", unsafe_allow_html=True)


# ── Main layout ────────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid #dde3ec; border-radius:10px;
                padding:1rem 1.6rem 0.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.04);
                margin-bottom:0.5rem;">
        <div style="display:flex; align-items:center; gap:8px;
                    border-bottom:1px solid #dde3ec; padding-bottom:0.75rem; margin-bottom:0.5rem;">
            <div style="width:9px; height:9px; border-radius:50%;
                        background:#0077b6; flex-shrink:0;"></div>
            <div style="{MONO} font-size:0.68rem; text-transform:uppercase;
                        letter-spacing:0.12em; color:#4a5568; font-weight:600;">
                Sensor Input &mdash; Current Turbine Telemetry
            </div>
        </div>
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

    st.markdown(f"""
    <div style="{MONO} font-size:0.67rem; color:#8896a8; text-transform:uppercase;
                letter-spacing:0.1em; margin:1rem 0 0.3rem;">
        Recent Temperature Trend (last 15 min)
    </div>""", unsafe_allow_html=True)

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

    st.markdown(f"""
    <div style="{MONO} font-size:0.63rem; color:#8896a8; margin:0.8rem 0 0.6rem; line-height:1.7;">
        Try these combinations to trigger different risk levels:<br>
        &nbsp;&nbsp;Temp &gt; 59C &rarr; Medium Risk &nbsp;|&nbsp;
        Temp &gt; 68C + rapidly rising &rarr; High Risk<br>
        &nbsp;&nbsp;High wind (&gt;4 m/s) + power &lt; 3 kW &rarr; Low/Medium Risk &nbsp;|&nbsp;
        Low power/wind ratio &rarr; Low Risk
    </div>
    """, unsafe_allow_html=True)

    st.button("Run Health Assessment")


# ── Right panel ────────────────────────────────────────────────────────────────
with right:
    label, proba = predict(wind, rotor, gen_speed, pitch, power, temp, temp_history)
    s    = RISK[label]
    conf = proba.max() * 100

    # Probability bars
    bars_html = ""
    for cls, p in zip(le.classes_, proba):
        pct   = p * 100
        w     = max(pct, 0.4)
        clr   = RISK[cls]["bar"]
        lbl_style = f"font-weight:600; color:{RISK[cls]['color']};" if cls == label else "color:#4a5568;"
        pct_style = f"font-weight:600; color:{RISK[cls]['color']};" if cls == label else "color:#0d1b2a;"
        bars_html += f"""
        <div style="display:flex; align-items:center; gap:9px; margin-bottom:6px;">
            <div style="{MONO} font-size:0.68rem; width:95px; flex-shrink:0; {lbl_style}">{cls}</div>
            <div style="flex:1; background:#eef2f7; border-radius:4px; height:8px; overflow:hidden;">
                <div style="height:100%; border-radius:4px; width:{w}%; background:{clr};"></div>
            </div>
            <div style="{MONO} font-size:0.68rem; width:36px; text-align:right;
                        flex-shrink:0; {pct_style}">{pct:.1f}%</div>
        </div>"""

    # Sensor tile helper
    def tile(name, value, unit, left_color="#00b4d8", val_color="#0d1b2a"):
        return f"""
        <div style="background:#f0f4f8; border:1px solid #dde3ec; border-radius:8px;
                    padding:10px 12px; border-left:3px solid {left_color};">
            <div style="{MONO} font-size:0.58rem; text-transform:uppercase;
                        letter-spacing:0.1em; color:#8896a8; margin-bottom:3px;">{name}</div>
            <div style="{MONO} font-size:1.05rem; font-weight:600; color:{val_color};">
                {value}<span style="{MONO} font-size:0.62rem; color:#8896a8; margin-left:2px;">{unit}</span>
            </div>
        </div>"""

    temp_color = "#b91c3a" if temp > 68 else "#b07d12" if temp > 58 else "#0d1b2a"
    temp_bd    = "#e11d48" if temp > 68 else "#f0a500" if temp > 58 else "#00b4d8"
    eff        = round(power / wind, 2) if wind > 0 else 0.0
    rec_title, rec_body = RECS[label]

    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid #dde3ec; border-radius:10px;
                padding:1.4rem 1.6rem; box-shadow:0 1px 3px rgba(0,0,0,0.04);">

        <div style="display:flex; align-items:center; gap:8px;
                    border-bottom:1px solid #dde3ec; padding-bottom:0.75rem; margin-bottom:1.2rem;">
            <div style="width:9px; height:9px; border-radius:50%;
                        background:{s['bar']}; flex-shrink:0;"></div>
            <div style="{MONO} font-size:0.68rem; text-transform:uppercase;
                        letter-spacing:0.12em; color:#4a5568; font-weight:600;">
                Health Assessment Output
            </div>
        </div>

        <div style="background:{s['bg']}; border:1px solid {s['bd']}; border-radius:10px;
                    padding:1.3rem 1.5rem; margin-bottom:1.2rem;">
            <div style="{MONO} font-size:0.62rem; text-transform:uppercase;
                        letter-spacing:0.15em; color:{s['color']}; opacity:0.75; margin-bottom:4px;">
                Risk Classification
            </div>
            <div style="{MONO} font-size:2rem; font-weight:700; letter-spacing:-0.02em;
                        line-height:1.1; color:{s['color']}; margin-bottom:3px;">
                {label.upper()}
            </div>
            <div style="{MONO} font-size:0.73rem; color:{s['color']}; opacity:0.65;">
                Confidence &nbsp;{conf:.1f}%
            </div>
        </div>

        <div style="{MONO} font-size:0.62rem; text-transform:uppercase;
                    letter-spacing:0.12em; color:#8896a8; margin-bottom:8px;">
            Class Probabilities
        </div>
        {bars_html}

        <div style="{MONO} font-size:0.62rem; text-transform:uppercase;
                    letter-spacing:0.12em; color:#8896a8; margin:1rem 0 0.5rem;">
            Live Readings
        </div>

        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:1rem;">
            {tile('Wind Speed', f'{wind:.1f}', 'm/s')}
            {tile('Rotor Speed', f'{rotor:.1f}', 'rpm')}
            {tile('Gen Speed', f'{gen_speed:.0f}', 'rpm')}
            {tile('Power Output', f'{power:.0f}', 'kW')}
            {tile('Efficiency', f'{eff:.2f}', 'kW/(m/s)')}
            {tile('Gen Temp', f'{temp:.1f}', 'C', left_color=temp_bd, val_color=temp_color)}
        </div>

        <div style="border-radius:0 8px 8px 0; border-left:3px solid {s['color']};
                    border-top:1px solid #dde3ec; border-right:1px solid #dde3ec;
                    border-bottom:1px solid #dde3ec; padding:0.9rem 1.1rem; background:#f0f4f8;">
            <div style="{SANS} font-size:0.82rem; font-weight:600;
                        color:#0d1b2a; margin-bottom:3px;">{rec_title}</div>
            <div style="{SANS} font-size:0.76rem; color:#4a5568; line-height:1.55;">{rec_body}</div>
        </div>

    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="{MONO} margin-top:2rem; padding:1rem 0; border-top:1px solid #dde3ec;
            display:flex; justify-content:space-between; font-size:0.63rem; color:#8896a8;">
    <span>Project Global-Grid &nbsp;&middot;&nbsp; Aventa AV-7 SCADA &nbsp;&middot;&nbsp;
          39M records &nbsp;&middot;&nbsp; Multi-Sensor Anomaly Scoring</span>
    <span>XGBoost &nbsp;&middot;&nbsp; SMOTE &nbsp;&middot;&nbsp;
          Macro F1: 0.97 &nbsp;&middot;&nbsp; Phase 10</span>
</div>
""", unsafe_allow_html=True)
