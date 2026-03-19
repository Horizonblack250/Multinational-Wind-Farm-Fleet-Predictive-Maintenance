import streamlit as st
import numpy as np
import joblib
import json
import os
import pandas as pd
import plotly.graph_objects as go

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
[data-testid="stTab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
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
    base       = os.path.join(os.path.dirname(__file__), "models")
    classifier = joblib.load(os.path.join(base, "polaris_risk_model.pkl"))
    encoder    = joblib.load(os.path.join(base, "polaris_label_encoder.pkl"))
    with open(os.path.join(base, "polaris_features.json"))  as f: features = json.load(f)
    with open(os.path.join(base, "degradation_stats.json")) as f: deg      = json.load(f)
    return classifier, encoder, features, deg

model, le, FEATURES, deg_data = load_artifacts()

RISK = {
    "Normal":      {"color": "#1b7f4f", "bg": "#e8f5ef", "bd": "#a8d5be", "bar": "#27ae60"},
    "Low Risk":    {"color": "#b07d12", "bg": "#fef9e7", "bd": "#f5d87a", "bar": "#f0a500"},
    "Medium Risk": {"color": "#c0420a", "bg": "#fef0e7", "bd": "#f5b98a", "bar": "#e86a2a"},
    "High Risk":   {"color": "#b91c3a", "bg": "#fdeef1", "bd": "#f5a0b0", "bar": "#e11d48"},
}
RECS = {
    "Normal":      ("All systems nominal",         "Operating within expected parameters. No action required. Continue standard monitoring interval."),
    "Low Risk":    ("Elevated readings detected",  "One or more sensors trending above baseline. Schedule a visual inspection within 72 hours."),
    "Medium Risk": ("Abnormal operating state",    "Multiple anomaly signals detected. Reduce load where possible. Dispatch inspection within 24 hours."),
    "High Risk":   ("Critical — immediate action", "Severe risk of turbine failure. Take turbine offline and conduct emergency inspection now."),
}
MONO = "font-family:'IBM Plex Mono',monospace;"
SANS = "font-family:'IBM Plex Sans',sans-serif;"

SCENARIO_CONTEXT = {
    "normal":  "Turbine running under light load in mild conditions. Generator temperature is low and stable — typical overnight or low-wind operation.",
    "warming": "Generator temperature is climbing steadily. This pattern often precedes a maintenance alert if the trend continues for another 30–60 minutes.",
    "high":    "Sustained operation at high temperature. The turbine has been running hot for over an hour. LSTM forecasts whether it will plateau or keep rising.",
    "cooling": "Temperature is falling — consistent with a load reduction or wind drop. Forecast confirms whether cooldown is proceeding normally.",
}


# ── Classifier ─────────────────────────────────────────────────────────────────
def classify(wind, rotor, gen_speed, pitch, power, temp, temp_history):
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


# ── Plotly forecast chart builder ──────────────────────────────────────────────
def build_forecast_chart(history_vals, forecast_vals):
    hist_x = list(range(-len(history_vals), 0))
    fc_x   = list(range(0, len(forecast_vals)))

    # Connect the gap: last historical point bridges into forecast
    bridge_x    = [hist_x[-1], fc_x[0]]
    bridge_vals = [history_vals[-1], forecast_vals[0]]

    all_vals   = history_vals + forecast_vals
    y_min      = min(all_vals) - 5
    y_max      = max(all_vals) + 8

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_x, y=history_vals,
        mode='lines',
        name='Historical',
        line=dict(color='#0077b6', width=2.5),
        hovertemplate='Minute %{x}<br>Temp: %{y:.1f}°C<extra></extra>'
    ))

    # Bridge connector (dashed, subtle)
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_vals,
        mode='lines',
        name='',
        line=dict(color='#94a3b8', width=1.5, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_x, y=forecast_vals,
        mode='lines',
        name='LSTM Forecast',
        line=dict(color='#e11d48', width=2.5, dash='dash'),
        hovertemplate='Minute +%{x}<br>Forecast: %{y:.1f}°C<extra></extra>'
    ))

    # Forecast uncertainty band (±2°C visual band)
    upper = [v + 2.0 for v in forecast_vals]
    lower = [v - 2.0 for v in forecast_vals]
    fig.add_trace(go.Scatter(
        x=fc_x + fc_x[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(225,29,72,0.08)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip',
        name='Uncertainty band'
    ))

    # Medium Risk threshold line at 58°C
    if y_max > 50:
        fig.add_hline(
            y=58,
            line_dash="dash",
            line_color="#e86a2a",
            line_width=1.5,
            annotation_text="Medium Risk threshold (58°C)",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#e86a2a"),
        )

    # High Risk threshold line at 68°C
    if y_max > 60:
        fig.add_hline(
            y=68,
            line_dash="dash",
            line_color="#b91c3a",
            line_width=1.5,
            annotation_text="High Risk threshold (68°C)",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#b91c3a"),
        )

    # "Now" vertical line at x=0
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="#334155",
        line_width=1.5,
        annotation_text="Now — forecast begins",
        annotation_position="top",
        annotation_font=dict(size=11, color="#334155"),
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#f8fafc',
        height=360,
        margin=dict(l=10, r=10, t=30, b=40),
        font=dict(family='IBM Plex Mono, monospace', size=11, color='#4a5568'),
        xaxis=dict(
            title='Minutes (negative = history, positive = forecast)',
            gridcolor='#e2e8f0',
            zeroline=False,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title='Generator Temperature (°C)',
            gridcolor='#e2e8f0',
            range=[y_min, y_max],
            tickfont=dict(size=10),
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='left', x=0,
            font=dict(size=11),
        ),
        hovermode='x unified',
    )

    return fig


# ── Hero ───────────────────────────────────────────────────────────────────────
fs = deg_data['fleet_stats']

st.markdown("""
<div style="background:linear-gradient(135deg,#0077b6 0%,#023e8a 60%,#03045e 100%);
            border-radius:0 0 16px 16px; padding:1.6rem 2.2rem 1.4rem;
            margin:0 0 1.4rem; display:flex; align-items:center;
            justify-content:space-between;">
    <div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    letter-spacing:0.2em; text-transform:uppercase; color:#90e0ef; margin-bottom:5px;">
            Wind Farm Predictive Analytics Platform
        </div>
        <div style="font-size:1.25rem; font-weight:700; color:#ffffff; line-height:1.3; margin-bottom:5px;">
            Project Global-Grid &nbsp;&middot;&nbsp;
            Predictive Maintenance of a Multi-National Wind Farm Fleet
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#90e0ef; opacity:0.85;">
            SCADA Telemetry &nbsp;|&nbsp; Aventa AV-7 &nbsp;|&nbsp;
            Dec 2021 &rarr; Jul 2023 &nbsp;|&nbsp; 712K operational minutes
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
            XGBoost &nbsp;&middot;&nbsp; LSTM &nbsp;&middot;&nbsp; Multi-Sensor Scoring
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Fleet KPIs ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Classifier Macro F1",   "0.98",         help="Across all 4 risk classes")
k2.metric("LSTM Forecast Horizon", "30 min",        help="Temperature trajectory prediction")
k3.metric("Avg Generator Temp",    f"{fs['overall_avg_temp']}°C",   help="Fleet average over full period")
k4.metric("Peak Temp Recorded",    f"{fs['peak_temp_recorded']}°C", help="Historical maximum")
k5.metric("High Risk Rate",        f"{fs['high_risk_pct']:.1f}%",   help="Of all operational minutes")

st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  Risk Monitor  ",
    "  Temp Forecast  ",
    "  Degradation Trends  ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Risk Monitor
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown(f"""
        <div style="{MONO} font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em;
                    color:#4a5568; font-weight:600; border-bottom:2px solid #0077b6;
                    padding-bottom:0.5rem; margin-bottom:1.2rem;">
            Sensor Input — Current Turbine Telemetry
        </div>""", unsafe_allow_html=True)

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
        <div style="{MONO} font-size:0.63rem; color:#8896a8; margin:0.8rem 0 0.6rem;
                    line-height:1.8; background:#f8fafc; border:1px solid #dde3ec;
                    border-radius:6px; padding:0.7rem 0.9rem;">
            <strong style="color:#4a5568;">Trigger signals:</strong><br>
            Temp &gt; 59C &rarr; Medium Risk &nbsp;|&nbsp;
            Temp &gt; 68C + rapidly rising &rarr; High Risk<br>
            High wind + power &lt; 3 kW &rarr; Low/Medium Risk &nbsp;|&nbsp;
            Low power/wind ratio &rarr; Low Risk
        </div>""", unsafe_allow_html=True)

        st.button("Run Health Assessment")

    with right:
        label, proba = classify(wind, rotor, gen_speed, pitch, power, temp, temp_history)
        s    = RISK[label]
        conf = proba.max() * 100

        st.markdown(f"""
        <div style="{MONO} font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em;
                    color:#4a5568; font-weight:600; border-bottom:2px solid {s['bar']};
                    padding-bottom:0.5rem; margin-bottom:1.2rem;">
            Health Assessment Output
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{s['bg']}; border:1px solid {s['bd']}; border-radius:10px;
                    padding:1.3rem 1.5rem; margin-bottom:1.2rem;">
            <div style="{MONO} font-size:0.62rem; text-transform:uppercase; letter-spacing:0.15em;
                        color:{s['color']}; opacity:0.75; margin-bottom:4px;">Risk Classification</div>
            <div style="{MONO} font-size:2rem; font-weight:700; color:{s['color']}; margin-bottom:3px;">
                {label.upper()}</div>
            <div style="{MONO} font-size:0.73rem; color:{s['color']}; opacity:0.65;">
                Confidence &nbsp;{conf:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="{MONO} font-size:0.62rem; text-transform:uppercase; letter-spacing:0.12em;
                    color:#8896a8; margin-bottom:0.5rem;">Class Probabilities</div>
        """, unsafe_allow_html=True)

        for cls, p in zip(le.classes_, proba):
            pct     = p * 100
            is_pred = cls == label
            lc      = RISK[cls]['color'] if is_pred else '#4a5568'
            fw      = '600' if is_pred else '400'
            ca, cb  = st.columns([3, 1])
            with ca:
                st.markdown(f"""
                <div style="{MONO} font-size:0.7rem; color:{lc}; font-weight:{fw}; margin-bottom:2px;">
                    {cls}</div>
                <div style="background:#eef2f7; border-radius:4px; height:8px;
                            overflow:hidden; margin-bottom:8px;">
                    <div style="height:100%; border-radius:4px; width:{max(pct,0.4)}%;
                                background:{RISK[cls]['bar']};"></div>
                </div>""", unsafe_allow_html=True)
            with cb:
                st.markdown(f"""
                <div style="{MONO} font-size:0.7rem; text-align:right; padding-top:2px;
                            color:{lc}; font-weight:{fw};">{pct:.1f}%</div>
                """, unsafe_allow_html=True)

        eff = round(power / wind, 2) if wind > 0 else 0.0
        temp_suffix = " (HIGH)" if temp > 68 else " (WARN)" if temp > 58 else ""

        st.markdown(f"""
        <div style="{MONO} font-size:0.62rem; text-transform:uppercase; letter-spacing:0.12em;
                    color:#8896a8; margin:1rem 0 0.5rem;">Live Readings</div>
        """, unsafe_allow_html=True)

        r1a, r1b, r1c = st.columns(3)
        r1a.metric("Wind Speed",   f"{wind:.1f} m/s")
        r1b.metric("Rotor Speed",  f"{rotor:.1f} rpm")
        r1c.metric("Gen Speed",    f"{gen_speed:.0f} rpm")
        r2a, r2b, r2c = st.columns(3)
        r2a.metric("Power Output", f"{power:.0f} kW")
        r2b.metric("Efficiency",   f"{eff:.2f} kW/ms")
        r2c.metric(f"Gen Temp{temp_suffix}", f"{temp:.1f} C")

        rec_title, rec_body = RECS[label]
        st.markdown(f"""
        <div style="border-left:3px solid {s['color']}; border-top:1px solid #dde3ec;
                    border-right:1px solid #dde3ec; border-bottom:1px solid #dde3ec;
                    border-radius:0 8px 8px 0; padding:0.9rem 1.1rem;
                    background:#f8fafc; margin-top:1rem;">
            <div style="{SANS} font-size:0.82rem; font-weight:600;
                        color:#0d1b2a; margin-bottom:3px;">{rec_title}</div>
            <div style="{SANS} font-size:0.76rem; color:#4a5568; line-height:1.55;">{rec_body}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LSTM Temperature Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="{MONO} font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em;
                color:#4a5568; font-weight:600; border-bottom:2px solid #0077b6;
                padding-bottom:0.5rem; margin-bottom:1rem;">
        30-Minute Temperature Forecast — LSTM Model
    </div>""", unsafe_allow_html=True)

    # Plain English explainer
    st.markdown(f"""
    <div style="{SANS} font-size:0.84rem; color:#4a5568; background:#f8fafc;
                border:1px solid #dde3ec; border-radius:8px; padding:0.9rem 1.1rem;
                margin-bottom:1.2rem; line-height:1.7;">
        <strong style="color:#0d1b2a;">How to read this chart:</strong>
        The <span style="color:#0077b6; font-weight:600;">blue line</span> shows the
        generator temperature over the last 60 minutes of real SCADA data.
        The <span style="color:#e11d48; font-weight:600;">red dashed line</span> is what
        the LSTM model predicts will happen over the next 30 minutes.
        The shaded red area is the uncertainty band (&plusmn;2&deg;C).
        Dashed horizontal lines mark the thresholds at which maintenance alerts trigger.
    </div>""", unsafe_allow_html=True)

    seqs = deg_data['representative_sequences']
    scenario_map = {
        "Normal operation (avg ~11°C)":        "normal",
        "Warming trend (rising temperature)":   "warming",
        "Sustained high temperature (~54°C)":   "high",
        "Cooling down (post-load reduction)":   "cooling",
    }

    sel = st.selectbox("Select operational scenario", list(scenario_map.keys()))
    key = scenario_map[sel]

    history_vals  = seqs[key]['history']
    forecast_vals = seqs[key]['forecast']
    last_temp     = history_vals[-1]
    max_fc        = max(forecast_vals)
    min_fc        = min(forecast_vals)
    trend_dir     = (
        "RISING"  if forecast_vals[-1] > forecast_vals[0] + 0.5 else
        "FALLING" if forecast_vals[-1] < forecast_vals[0] - 0.5 else
        "STABLE"
    )

    # Scenario context card
    st.markdown(f"""
    <div style="background:#e8f0fe; border:1px solid #bfdbfe; border-radius:8px;
                padding:0.8rem 1.1rem; margin-bottom:1rem;">
        <div style="{MONO} font-size:0.6rem; text-transform:uppercase; letter-spacing:0.12em;
                    color:#1e40af; margin-bottom:3px;">Scenario context</div>
        <div style="{SANS} font-size:0.82rem; color:#1e3a8a;">
            {SCENARIO_CONTEXT[key]}
        </div>
    </div>""", unsafe_allow_html=True)

    # Forecast KPIs
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Current Temp",  f"{last_temp:.1f}°C")
    fc2.metric("Forecast Peak", f"{max_fc:.1f}°C")
    fc3.metric("Forecast Low",  f"{min_fc:.1f}°C")
    fc4.metric("30-min Trend",  trend_dir)

    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    # Plotly chart
    fig = build_forecast_chart(history_vals, forecast_vals)
    st.plotly_chart(fig, use_container_width=True)

    # Alert interpretation
    alert_color = "#b91c3a" if max_fc > 68 else "#c0420a" if max_fc > 58 else "#1b7f4f"
    alert_bg    = "#fdeef1" if max_fc > 68 else "#fef0e7" if max_fc > 58 else "#e8f5ef"
    alert_text  = (
        "CRITICAL: Forecast exceeds the High Risk threshold (68°C). "
        "Recommend immediate load reduction and unscheduled inspection."
        if max_fc > 68 else
        "WARNING: Forecast approaches the Medium Risk threshold (58°C). "
        "Continue monitoring — if trend persists, schedule inspection within 24 hours."
        if max_fc > 58 else
        "Forecast remains within the normal operating envelope. "
        "No maintenance action required based on this trajectory."
    )

    st.markdown(f"""
    <div style="border-left:3px solid {alert_color}; background:{alert_bg};
                border-top:1px solid {alert_color}33; border-right:1px solid {alert_color}33;
                border-bottom:1px solid {alert_color}33;
                border-radius:0 8px 8px 0; padding:0.9rem 1.1rem; margin-top:0.2rem;">
        <div style="{MONO} font-size:0.62rem; text-transform:uppercase; letter-spacing:0.1em;
                    color:{alert_color}; margin-bottom:4px;">Forecast Interpretation</div>
        <div style="{SANS} font-size:0.82rem; color:#0d1b2a; font-weight:500;">{alert_text}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Degradation Trends
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="{MONO} font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em;
                color:#4a5568; font-weight:600; border-bottom:2px solid #0077b6;
                padding-bottom:0.5rem; margin-bottom:0.3rem;">
        Turbine Health Degradation — Dec 2021 to Jul 2023
    </div>
    <div style="{SANS} font-size:0.82rem; color:#4a5568; margin-bottom:1.2rem; line-height:1.6;">
        Long-term trends reveal gradual degradation patterns invisible in real-time monitoring.
        Rising baseline temperature or falling efficiency over weeks signals developing faults
        before they trigger acute risk alerts.
    </div>""", unsafe_allow_html=True)

    weekly = pd.DataFrame(deg_data['weekly_trends'])
    weekly['date'] = pd.to_datetime(weekly['date'])

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Data Range",           f"{fs['date_range_start']} → {fs['date_range_end']}")
    d2.metric("Operational Minutes",   f"{fs['total_minutes']:,}")
    d3.metric("Avg Efficiency",        f"{fs['overall_avg_eff']:.3f} kW/ms")
    d4.metric("High Risk Minutes",     f"{fs['high_risk_pct']:.2f}%")

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    view = st.radio(
        "Select trend view",
        ["Generator Temperature Drift", "Efficiency Degradation", "Risk Event Frequency"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    if view == "Generator Temperature Drift":
        chart_data = weekly.set_index('date')[['avg_temp', 'temp_trend']].copy()
        chart_data.columns = ['Daily avg temp (°C)', '7-day rolling avg (°C)']
        st.line_chart(chart_data, color=["#90caf9", "#0077b6"], height=320)
        st.markdown(f"""
        <div style="{MONO} font-size:0.65rem; color:#8896a8; text-align:center; margin-top:0.3rem;">
            Light blue = daily average &nbsp;|&nbsp; Dark blue = 7-day rolling average
        </div>""", unsafe_allow_html=True)
        first_month = weekly.head(30)['avg_temp'].mean()
        last_month  = weekly.tail(30)['avg_temp'].mean()
        drift       = last_month - first_month
        drift_color = "#b91c3a" if drift > 3 else "#c0420a" if drift > 1 else "#1b7f4f"
        drift_msg   = (
            f"Temperature baseline has {'risen' if drift > 0 else 'fallen'} "
            f"{abs(drift):.1f}°C over the monitoring period — "
            f"{'significant thermal degradation detected.' if drift > 3 else 'moderate upward drift, monitor closely.' if drift > 1 else 'within acceptable variance.'}"
        )

    elif view == "Efficiency Degradation":
        chart_data = weekly.set_index('date')[['avg_eff', 'eff_trend']].copy()
        chart_data.columns = ['Daily efficiency (kW/ms)', '7-day rolling avg (kW/ms)']
        st.line_chart(chart_data, color=["#a5d6a7", "#2d6a4f"], height=320)
        st.markdown(f"""
        <div style="{MONO} font-size:0.65rem; color:#8896a8; text-align:center; margin-top:0.3rem;">
            Light green = daily efficiency &nbsp;|&nbsp; Dark green = 7-day rolling average
        </div>""", unsafe_allow_html=True)
        first_eff = weekly.head(30)['avg_eff'].mean()
        last_eff  = weekly.tail(30)['avg_eff'].mean()
        eff_drop  = ((first_eff - last_eff) / first_eff * 100) if first_eff > 0 else 0
        drift_color = "#b91c3a" if eff_drop > 15 else "#c0420a" if eff_drop > 5 else "#1b7f4f"
        drift_msg = (
            f"Turbine efficiency has {'dropped' if eff_drop > 0 else 'improved'} "
            f"{abs(eff_drop):.1f}% over the monitoring period — "
            f"{'significant mechanical degradation.' if eff_drop > 15 else 'moderate decline, schedule maintenance review.' if eff_drop > 5 else 'performance stable.'}"
        )

    else:
        chart_data = weekly.set_index('date')[['high_risk_pct', 'med_risk_pct']].copy()
        chart_data.columns = ['High Risk %', 'Medium Risk %']
        st.line_chart(chart_data, color=["#e11d48", "#e86a2a"], height=320)
        st.markdown(f"""
        <div style="{MONO} font-size:0.65rem; color:#8896a8; text-align:center; margin-top:0.3rem;">
            Red = High Risk minutes per day (%) &nbsp;|&nbsp; Orange = Medium Risk minutes per day (%)
        </div>""", unsafe_allow_html=True)
        avg_hr     = weekly['high_risk_pct'].mean()
        peak_hr    = weekly['high_risk_pct'].max()
        drift_color = "#b91c3a" if peak_hr > 5 else "#c0420a" if peak_hr > 1 else "#1b7f4f"
        drift_msg  = (
            f"Average High Risk event rate: {avg_hr:.2f}% of daily minutes. "
            f"Peak recorded: {peak_hr:.2f}% in a single day. "
            f"{'Elevated event clustering detected — review operational logs.' if peak_hr > 5 else 'Event rate within expected bounds for this turbine class.'}"
        )

    st.markdown(f"""
    <div style="border-left:3px solid {drift_color}; border-top:1px solid #dde3ec;
                border-right:1px solid #dde3ec; border-bottom:1px solid #dde3ec;
                border-radius:0 8px 8px 0; padding:0.9rem 1.1rem;
                background:#f8fafc; margin-top:0.8rem;">
        <div style="{MONO} font-size:0.62rem; text-transform:uppercase; letter-spacing:0.1em;
                    color:{drift_color}; margin-bottom:4px;">Trend Analysis</div>
        <div style="{SANS} font-size:0.82rem; color:#0d1b2a; font-weight:500;">{drift_msg}</div>
    </div>""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="{MONO} margin-top:2rem; padding:1rem 0; border-top:1px solid #dde3ec;
            display:flex; justify-content:space-between; font-size:0.63rem; color:#8896a8;">
    <span>Project Global-Grid &nbsp;&middot;&nbsp; Aventa AV-7 SCADA
          &nbsp;&middot;&nbsp; 39M raw records &nbsp;&middot;&nbsp; 712K operational minutes</span>
    <span>XGBoost Classifier &nbsp;&middot;&nbsp; LSTM Forecaster
          &nbsp;&middot;&nbsp; Macro F1: 0.98 &nbsp;&middot;&nbsp; Phase 10</span>
</div>
""", unsafe_allow_html=True)
