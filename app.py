"""
CyberSentinel — Threat Intelligence Dashboard (Streamlit)
============================================================
Volumetric forecasting (Holt-Winters/ARIMA via statsmodels) and behavioral
anomaly detection (Isolation Forest via scikit-learn), fit live on the
underlying data rather than replaying a pre-generated JSON file.

Run locally:    streamlit run app.py
Run in Docker:  docker compose up --build
"""

import os

# Must run before numpy/pandas/sklearn/statsmodels are imported: on constrained
# cloud containers (e.g. Streamlit Community Cloud), the underlying BLAS
# library and joblib's multiprocessing backend can spawn more native
# threads/workers than the container's CPU quota actually allows. That thread
# oversubscription is a known cause of hard segmentation faults during model
# fitting (IsolationForest, ARIMA) — a C-level crash no Python try/except can
# catch. Pinning these to 1 keeps everything single-threaded and avoids it.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).parent


def _find_data_file(filename: str) -> Path:
    """Look in data/ first, then fall back to the repo root (handles a
    flat GitHub 'Add files via upload' layout)."""
    for candidate in (APP_DIR / "data" / filename, APP_DIR / filename):
        if candidate.exists():
            return candidate
    return APP_DIR / "data" / filename


FORECAST_CSV = _find_data_file("threat_forecast.csv")
INCIDENTS_CSV = _find_data_file("live_incidents.csv")

CHARCOAL = "#1A1D20"
CRIMSON = "#D9534F"
SAGE = "#6E8E75"
AMBER = "#E29578"
BG_CREAM = "#F9F6F0"
MANILA = "#E8DCC3"
HAIRLINE = "#DDD5C4"
CHART_TEMPLATE = "plotly_white"

# Fixed light-theme colors. The app previously used Streamlit's theme CSS
# variables (var(--text-color) etc.), which follow whatever light/dark mode
# the viewer's browser or Streamlit "Settings" menu selects. That's what
# caused the inconsistent switching — on some viewers the theme resolved to
# dark, clashing with the cream/manila palette. Using fixed hex values here
# instead means the page always renders the same way regardless of the
# viewer's theme setting. Paired with .streamlit/config.toml (theme.base =
# "light") and client.toolbarMode = "minimal" (which hides the Settings
# menu's theme toggle), this locks the app to a single permanent light theme.
PAGE_TEXT = CHARCOAL
PAGE_BG = BG_CREAM
PAGE_SECONDARY_BG = MANILA

# Traffic Light Protocol (TLP) — the real convention security analysts use to
# mark how widely a piece of intelligence may be shared. Used here as the
# page-level classification banner instead of a generic page subtitle.
TLP = {
    "RED":   {"color": CRIMSON, "label": "TLP:RED",   "desc": "Not for disclosure — restricted to named recipients"},
    "AMBER": {"color": AMBER,   "label": "TLP:AMBER", "desc": "Limited disclosure — internal use only"},
    "GREEN": {"color": SAGE,    "label": "TLP:GREEN", "desc": "Community disclosure — sector peers"},
}

REQUIRED_FORECAST_COLS = {"ds", "y"}
REQUIRED_INCIDENT_COLS = {
    "timestamp", "host", "payload_size_kb", "response_latency_ms",
    "failed_auth_count", "bytes_out_mb", "request_rate", "severity",
    "event_type", "source_ip", "destination_port",
}

# --------------------------------------------------------------------------
# Page config & styling — analyst case-file aesthetic: cream/manila paper,
# a serif "field report" display face, JetBrains Mono for raw data, and a
# TLP classification banner as the recurring signature element.
# --------------------------------------------------------------------------
st.set_page_config(page_title="CyberSentinel — Threat Intelligence", page_icon="🛡️", layout="wide")

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

        /* Scope the display font to text elements only — a blanket rule here
           would also override Streamlit's icon elements, which render as an
           icon only because the browser uses a ligature font (Material
           Symbols) to turn their literal text content (e.g.
           "keyboard_double_arrow_left") into a glyph. Overriding that font
           breaks the ligature and leaves the raw icon name showing as text. */
        [data-testid="stAppViewContainer"] *:not([data-testid^="stIcon"]) {{
            font-family: 'Inter', sans-serif;
        }}
        [data-testid^="stIcon"] {{ font-family: 'Material Symbols Rounded' !important; }}
        .block-container {{ padding-top: 1.5rem; }}

        /* Hard-lock the light background across the app body and sidebar,
           so the page can't render dark even if a viewer's browser or the
           deployment environment ignores config.toml's theme setting. */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: {PAGE_BG} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {PAGE_SECONDARY_BG} !important;
        }}
        [data-testid="stAppViewContainer"] * {{
            color: {PAGE_TEXT};
        }}

        /* Fixed light-theme colors throughout (see PAGE_TEXT / PAGE_BG /
           PAGE_SECONDARY_BG above) rather than Streamlit's theme CSS
           variables, so the page never flips to a dark palette. Only the
           case-file accent colors (crimson/sage/amber) were already fixed,
           since they're mid-tone enough to read on a light surface. */
        h1 {{
            font-family: 'Source Serif 4', Georgia, serif !important;
            font-weight: 700 !important;
            color: {PAGE_TEXT} !important;
            letter-spacing: -0.01em;
            border-bottom: 3px solid {PAGE_TEXT};
            padding-bottom: 10px;
        }}
        h2 {{ font-family: 'Source Serif 4', Georgia, serif !important; color: {PAGE_TEXT} !important; font-weight: 700 !important; }}
        h3 {{
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            font-size: 0.85rem !important;
            letter-spacing: 0.06em;
            color: {PAGE_TEXT} !important;
            border-left: 3px solid {CRIMSON};
            padding-left: 10px;
            margin-top: 1.6rem !important;
        }}

        div[data-testid="stMetric"] {{
            background: {PAGE_SECONDARY_BG};
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 8px; padding: 14px 18px;
        }}
        div[data-testid="stMetricLabel"] {{ color: {PAGE_TEXT} !important; opacity: 0.65; font-family: 'Inter', sans-serif; }}
        div[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; color: {PAGE_TEXT} !important; }}

        .subtle {{ color: {PAGE_TEXT}; opacity: 0.65; font-size: 0.9rem; }}

        .sev-badge {{
            display: inline-block; padding: 2px 10px; border-radius: 3px; font-size: 0.72rem; margin-right: 6px;
            font-family: 'JetBrains Mono', monospace; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
            border: 1px solid currentColor;
        }}
        .sev-CRITICAL {{ background: color-mix(in srgb, {CRIMSON} 15%, {PAGE_BG}); color: {CRIMSON}; }}
        .sev-HIGH {{ background: color-mix(in srgb, {AMBER} 15%, {PAGE_BG}); color: {AMBER}; }}
        .sev-MEDIUM {{ background: color-mix(in srgb, {SAGE} 15%, {PAGE_BG}); color: {SAGE}; }}
        .sev-LOW {{ background: {PAGE_SECONDARY_BG}; color: {PAGE_TEXT}; opacity: 0.75; }}

        .tlp-banner {{
            display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700;
            letter-spacing: 0.08em; text-transform: uppercase;
            padding: 8px 14px; border-radius: 3px; margin: 2px 0 22px 0;
        }}
        .tlp-desc {{ font-weight: 400; letter-spacing: 0.02em; text-transform: none; opacity: 0.85; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def classification_banner(level: str):
    """Render the TLP classification banner — the page's signature element."""
    s = TLP[level]
    bg = f"color-mix(in srgb, {s['color']} 15%, {PAGE_BG})"
    st.markdown(
        f'<div class="tlp-banner" style="background:{bg};color:{s["color"]};'
        f'border-left:4px solid {s["color"]};">{s["label"]}'
        f'<span class="tlp-desc">— {s["desc"]}</span></div>',
        unsafe_allow_html=True,
    )




# --------------------------------------------------------------------------
# Sidebar — data source
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ CyberSentinel")
st.sidebar.markdown(
    '<span class="subtle">Live volumetric forecasting & behavioral anomaly '
    "detection over network security data.</span>",
    unsafe_allow_html=True,
)
st.sidebar.divider()
st.sidebar.markdown("### Data source")
forecast_upload = st.sidebar.file_uploader("Weekly traffic CSV (ds, y)", type="csv", key="forecast_upload")
incidents_upload = st.sidebar.file_uploader("Security incidents CSV", type="csv", key="incidents_upload")
with st.sidebar.expander("Expected CSV columns"):
    st.markdown(
        "**Weekly traffic:** `ds` (date), `y` (numeric volume)\n\n"
        "**Incidents:** `timestamp, host, payload_size_kb, response_latency_ms, "
        "failed_auth_count, bytes_out_mb, request_rate, severity, event_type, "
        "source_ip, destination_port`"
    )


def validate_csv(df: pd.DataFrame, required: set, label: str) -> bool:
    missing = required - set(df.columns)
    if missing:
        st.sidebar.error(f"{label} missing column(s): {', '.join(sorted(missing))}")
        return False
    return True


def _read_uploaded_csv(uploaded_file):
    """Read an st.file_uploader object safely across script reruns.

    Streamlit reruns the whole script on every interaction (including just
    switching sidebar pages), but keeps handing back the *same* underlying
    file object rather than a fresh one. Once pandas has read it, its
    position sits at EOF — the next rerun's pd.read_csv() then raises
    EmptyDataError and crashes the app. Seeking back to 0 first fixes that.
    """
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@st.cache_data
def load_bundled():
    missing = [p for p in (FORECAST_CSV, INCIDENTS_CSV) if not p.exists()]
    if missing:
        st.error(
            "**Bundled data files are missing from this deployment.**\n\n"
            + "\n".join(f"- Expected but not found: `{p}`" for p in missing)
            + "\n\nCheck that these files were committed and pushed to your repo "
            "(`git ls-files data/`). In the meantime, upload your own CSVs in the sidebar."
        )
        return pd.DataFrame(columns=["ds", "y"]), pd.DataFrame(columns=list(REQUIRED_INCIDENT_COLS))
    return pd.read_csv(FORECAST_CSV), pd.read_csv(INCIDENTS_CSV)


bundled_forecast, bundled_incidents = load_bundled()

if forecast_upload is not None:
    fc_raw = _read_uploaded_csv(forecast_upload)
    if fc_raw.empty:
        st.sidebar.error("Weekly traffic CSV appears empty or unreadable — falling back to bundled data.")
        fc_raw = bundled_forecast
    elif validate_csv(fc_raw, REQUIRED_FORECAST_COLS, "Weekly traffic CSV"):
        st.sidebar.success(f"Using uploaded traffic data ({len(fc_raw)} rows).")
    else:
        fc_raw = bundled_forecast
else:
    fc_raw = bundled_forecast

if incidents_upload is not None:
    inc_raw = _read_uploaded_csv(incidents_upload)
    if inc_raw.empty:
        st.sidebar.error("Incidents CSV appears empty or unreadable — falling back to bundled data.")
        inc_raw = bundled_incidents
    elif validate_csv(inc_raw, REQUIRED_INCIDENT_COLS, "Incidents CSV"):
        st.sidebar.success(f"Using uploaded incidents data ({len(inc_raw)} rows).")
    else:
        inc_raw = bundled_incidents
else:
    inc_raw = bundled_incidents

if forecast_upload is None and incidents_upload is None:
    st.sidebar.caption("Currently showing bundled sample data.")

if fc_raw.empty and inc_raw.empty:
    st.warning("No data available — upload CSVs in the sidebar to get started.")
    st.stop()


@st.cache_data
def process_data(fc_raw: pd.DataFrame, inc_raw: pd.DataFrame):
    fc = fc_raw.copy()
    fc["ds"] = pd.to_datetime(fc["ds"])
    fc = fc.sort_values("ds").reset_index(drop=True)

    inc = inc_raw.copy()
    if not inc.empty:
        inc["timestamp"] = pd.to_datetime(inc["timestamp"])
        inc = inc.sort_values("timestamp").reset_index(drop=True)
    return fc, inc


forecast_df, incidents_df = process_data(fc_raw, inc_raw)

st.sidebar.divider()
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Endpoints", "Predictive Engine", "Threat Hunting"],
    label_visibility="collapsed",
    key="page_nav",
)
st.sidebar.divider()
st.sidebar.caption("Forecasting: statsmodels (Holt-Winters / ARIMA) · Anomaly detection: scikit-learn IsolationForest")


# --------------------------------------------------------------------------
# ML: live forecasting
# --------------------------------------------------------------------------
def fit_holt_winters(series: np.ndarray, horizon: int):
    model = ExponentialSmoothing(series, trend="add", damped_trend=True, seasonal=None)
    fit = model.fit(optimized=True)
    return fit.forecast(horizon), series - fit.fittedvalues


def fit_arima(series: np.ndarray, horizon: int, order=(1, 1, 1)):
    fit = ARIMA(series, order=order).fit()
    return fit.forecast(horizon), fit.resid


def rmse(a, p):
    return float(np.sqrt(np.mean((np.array(a) - np.array(p)) ** 2)))


def mape(a, p):
    a, p = np.array(a), np.array(p)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


@st.cache_data
def run_forecast(fc: pd.DataFrame, horizon: int, model_name: str):
    y = fc["y"].astype(float).values
    holdout = min(8, max(2, len(y) // 5))
    train, test = y[:-holdout], y[-holdout:]
    fitter = fit_holt_winters if model_name == "Holt-Winters" else fit_arima

    test_preds, _ = fitter(train, holdout)
    metrics = {"rmse": rmse(test, test_preds), "mape": mape(test, test_preds)}

    full_fc, full_resid = fitter(y, horizon)
    std = np.std(full_resid) if len(full_resid) else 0.0
    upper, lower = full_fc + 1.96 * std, np.maximum(full_fc - 1.96 * std, 0)

    last_date = fc["ds"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")
    return future_dates, full_fc, upper, lower, metrics


# --------------------------------------------------------------------------
# ML: live anomaly detection
# --------------------------------------------------------------------------
def classify_anomaly(row) -> str:
    if row["failed_auth_count"] > 20:
        return "Credential Spraying Spike"
    if row["payload_size_kb"] > 10000 and row["bytes_out_mb"] > 1000:
        return "Potential Data Exfiltration Drop"
    if row["response_latency_ms"] > 3000 and row["payload_size_kb"] > 5000:
        return "C2 Beacon Anomaly"
    if row["request_rate"] > 800:
        return "DDoS Surge Pattern"
    return "Multi-Vector Statistical Outlier"


def risk_score(row) -> int:
    s = abs(float(row["anomaly_score"]))
    if row["severity"] == "CRITICAL":
        return min(99, int(s * 180 + 70))
    if row["severity"] == "HIGH":
        return min(85, int(s * 140 + 50))
    return min(65, int(s * 100 + 30))


@st.cache_data
def run_isolation_forest(inc: pd.DataFrame, contamination: float):
    if inc.empty or len(inc) < 10:
        return pd.DataFrame()
    feature_cols = ["payload_size_kb", "response_latency_ms", "failed_auth_count", "bytes_out_mb", "request_rate"]
    features = inc[feature_cols].fillna(0)
    X = StandardScaler().fit_transform(features)

    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100, n_jobs=1)
    preds = clf.fit_predict(X)
    scores = clf.score_samples(X)

    df = inc.copy()
    df["anomaly_score"] = scores
    df["is_anomaly"] = preds == -1
    anomalies = df[df["is_anomaly"]].copy()
    if anomalies.empty:
        return anomalies
    anomalies["classification"] = anomalies.apply(classify_anomaly, axis=1)
    anomalies["risk_score"] = anomalies.apply(risk_score, axis=1)
    return anomalies.sort_values("risk_score", ascending=False)


# --------------------------------------------------------------------------
# Derived: endpoint node profiles & KPIs
# --------------------------------------------------------------------------
def build_endpoint_profiles(inc: pd.DataFrame) -> dict:
    profiles = {}
    for host, grp in inc.groupby("host"):
        grp = grp.sort_values("timestamp")
        critical = int((grp["severity"] == "CRITICAL").sum())
        high = int((grp["severity"] == "HIGH").sum())
        risk = min(99, critical * 22 + high * 12 + len(grp) * 2)
        profiles[host] = {
            "risk_score": risk,
            "status": "CRITICAL" if risk > 70 else "WARNING" if risk > 40 else "NOMINAL",
            "total_events": len(grp),
            "critical_events": critical,
            "data": grp,
        }
    return profiles


def compute_kpis(inc: pd.DataFrame, anomalies: pd.DataFrame) -> dict:
    total = len(inc)
    critical = int((inc["severity"] == "CRITICAL").sum()) if total else 0
    high = int((anomalies["severity"] == "HIGH").sum()) if not anomalies.empty else 0
    blocked = int(inc.loc[inc["severity"].isin(["CRITICAL", "HIGH"]), "payload_size_kb"].sum()) if total else 0
    active_mitigations = critical * 3 + high
    spi = max(0.0, min(100.0, 100 - (critical / max(total, 1)) * 80 - len(anomalies) * 0.6))
    anomaly_rate = round(len(anomalies) / max(total, 1) * 100, 2)
    return {
        "spi": round(spi, 1), "blocked_payloads": blocked, "active_mitigations": active_mitigations,
        "anomaly_rate": anomaly_rate, "total_events": total, "critical_events": critical,
    }


SEV_COLORS = {"CRITICAL": CRIMSON, "HIGH": AMBER, "MEDIUM": SAGE, "LOW": "#8A8474"}


def sev_badge(sev: str) -> str:
    return f'<span class="sev-badge sev-{sev}">{sev}</span>'


# ==========================================================================
# PAGE: Overview
# ==========================================================================
def render_overview():
    st.title("Security Posture Overview")
    classification_banner("AMBER")
    st.caption("Live KPIs computed from current incident data and anomaly detection.")

    contamination = st.sidebar.slider("Anomaly sensitivity (contamination)", 0.05, 0.30, 0.12, 0.01,
                                       key="overview_contam")
    anomalies = run_isolation_forest(incidents_df, contamination)
    kpis = compute_kpis(incidents_df, anomalies)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Security Posture Index", f"{kpis['spi']}", help="0–100, higher is safer")
    c2.metric("Blocked Payload Volume (KB)", f"{kpis['blocked_payloads']:,}")
    c3.metric("Active Mitigations", kpis["active_mitigations"])
    c4.metric("Anomaly Rate", f"{kpis['anomaly_rate']}%")

    st.markdown("### Incident Volume by Severity")
    if not incidents_df.empty:
        sev_counts = incidents_df["severity"].value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0)
        fig = px.bar(x=sev_counts.index, y=sev_counts.values, color=sev_counts.index,
                     color_discrete_map=SEV_COLORS, labels={"x": "Severity", "y": "Events"})
        fig.update_layout(template=CHART_TEMPLATE, height=340, showlegend=False,
                           margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, width="stretch", key="chart_severity_volume")

        st.markdown("### Traffic Volume by Host (derived from incidents)")
        host_traffic = incidents_df.groupby("host").agg(
            total_payload_kb=("payload_size_kb", "sum"), total_bytes_out_mb=("bytes_out_mb", "sum"),
            events=("host", "count"),
        ).round(1).sort_values("total_payload_kb", ascending=False)
        fig2 = px.bar(host_traffic.reset_index(), x="host", y="total_payload_kb", color="host",
                      color_discrete_sequence=[CRIMSON, AMBER, SAGE, CHARCOAL])
        fig2.update_layout(template=CHART_TEMPLATE, height=340, showlegend=False,
                            margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig2, width="stretch", key="chart_host_traffic")
        st.dataframe(host_traffic, width="stretch", key="table_host_traffic")
    else:
        st.info("No incident data loaded yet.")


# ==========================================================================
# PAGE: Endpoints
# ==========================================================================
def render_endpoints():
    st.title("Endpoint Deep-Dive")
    classification_banner("AMBER")
    st.caption("Per-host traffic history, latency, and raw event log.")

    if incidents_df.empty:
        st.info("No incident data loaded yet — upload a CSV in the sidebar.")
    else:
        profiles = build_endpoint_profiles(incidents_df)
        host = st.selectbox("Select host", sorted(profiles.keys()), key="endpoint_host_select")
        p = profiles[host]

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Risk Score", p["risk_score"])
        b2.metric("Status", p["status"])
        b3.metric("Total Events", p["total_events"])
        b4.metric("Critical Events", p["critical_events"])

        grp = p["data"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grp["timestamp"], y=grp["payload_size_kb"], name="Payload (KB)",
                                  line=dict(color=CRIMSON)))
        fig.add_trace(go.Scatter(x=grp["timestamp"], y=grp["response_latency_ms"], name="Latency (ms)",
                                  yaxis="y2", line=dict(color=SAGE, dash="dot")))
        fig.update_layout(
            template=CHART_TEMPLATE, height=380, margin=dict(t=20, l=10, r=10, b=10),
            yaxis=dict(title="Payload (KB)"), yaxis2=dict(title="Latency (ms)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, width="stretch", key="chart_endpoint_timeseries")

        st.markdown("### Raw event log")
        log = grp.sort_values("timestamp", ascending=False)[
            ["timestamp", "event_type", "severity", "source_ip", "destination_port", "failed_auth_count"]
        ].head(15)
        st.dataframe(log, width="stretch", hide_index=True, key="table_endpoint_log")


# ==========================================================================
# PAGE: Predictive Engine
# ==========================================================================
def render_predictive_engine():
    st.title("Predictive Engine")
    classification_banner("GREEN")
    st.caption("Live volumetric traffic forecast — fit on the fly from weekly history.")

    if forecast_df.empty or len(forecast_df) < 10:
        st.info("Not enough weekly traffic data to forecast yet — upload a CSV with `ds, y` columns.")
    else:
        col1, col2 = st.columns([1, 1])
        model_name = col1.radio("Model", ["Holt-Winters", "ARIMA"], horizontal=True, key="forecast_model_choice")
        horizon = col2.slider("Forecast horizon (weeks)", 4, 24, 16, key="forecast_horizon")

        future_dates, fc_vals, upper, lower, metrics = run_forecast(forecast_df, horizon, model_name)

        m1, m2 = st.columns(2)
        m1.metric("Backtest RMSE", f"{metrics['rmse']:,.0f}")
        m2.metric("Backtest MAPE", f"{metrics['mape']:.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["y"], name="Actual",
                                  line=dict(color=CHARCOAL, width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=fc_vals, name=f"{model_name} forecast",
                                  line=dict(color=CRIMSON, dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself", fillcolor="rgba(217,83,79,0.12)", line=dict(width=0),
            name="95% CI",
        ))
        fig.update_layout(template=CHART_TEMPLATE, height=440, margin=dict(t=20, l=10, r=10, b=10),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch", key="chart_forecast")

        with st.expander("View forecast table"):
            tbl = pd.DataFrame({"date": future_dates, "forecast": fc_vals.round(0),
                                 "lower_95": lower.round(0), "upper_95": upper.round(0)})
            st.dataframe(tbl, width="stretch", hide_index=True, key="table_forecast")


# ==========================================================================
# PAGE: Threat Hunting
# ==========================================================================
def render_threat_hunting():
    st.title("Threat Hunting")
    classification_banner("RED")
    st.caption("Behavioral anomalies detected live via Isolation Forest, classified by pattern.")

    if incidents_df.empty or len(incidents_df) < 10:
        st.info("Not enough incident data to run anomaly detection yet.")
    else:
        contamination = st.slider("Anomaly sensitivity (contamination)", 0.05, 0.30, 0.12, 0.01, key="hunt_contam")
        anomalies = run_isolation_forest(incidents_df, contamination)

        if anomalies.empty:
            st.success("No anomalies detected at this sensitivity level.")
        else:
            classes = sorted(anomalies["classification"].unique())
            selected = st.multiselect("Filter by classification", classes, default=classes, key="hunt_class_filter")
            filtered = anomalies[anomalies["classification"].isin(selected)]

            c1, c2, c3 = st.columns(3)
            c1.metric("Anomalies Detected", len(filtered))
            c2.metric("Critical Severity", int((filtered["severity"] == "CRITICAL").sum()))
            c3.metric("Avg Risk Score", f"{filtered['risk_score'].mean():.0f}" if len(filtered) else "—")

            fig = px.bar(filtered["classification"].value_counts().reset_index(),
                         x="classification", y="count", color="classification",
                         color_discrete_sequence=[CRIMSON, AMBER, SAGE, CHARCOAL, "#8A8474"])
            fig.update_layout(template=CHART_TEMPLATE, height=320, showlegend=False,
                               margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch", key="chart_anomaly_classes")

            st.markdown("### Anomaly Log")
            display = filtered[[
                "timestamp", "host", "classification", "severity", "risk_score",
                "payload_size_kb", "response_latency_ms", "failed_auth_count",
                "bytes_out_mb", "source_ip", "event_type",
            ]].reset_index(drop=True)
            st.dataframe(display, width="stretch", hide_index=True, key="table_anomaly_log")


PAGE_RENDERERS = {
    "Overview": render_overview,
    "Endpoints": render_endpoints,
    "Predictive Engine": render_predictive_engine,
    "Threat Hunting": render_threat_hunting,
}

# Each page renders inside its own try/except so that an error on one page
# (e.g. from an unusual uploaded CSV) shows as an inline message rather than
# crashing the whole app on the next rerun triggered by switching pages.
try:
    PAGE_RENDERERS[page]()
except Exception as e:
    st.error(
        f"Something went wrong rendering the **{page}** page: `{type(e).__name__}: {e}`\n\n"
        "Try switching to another page and back, or check the uploaded CSV. "
        "If this keeps happening, please share this exact message so it can be fixed."
    )
    with st.expander("Full error details"):
        st.exception(e)

st.sidebar.divider()
st.sidebar.caption("Built with Streamlit · scikit-learn · statsmodels · Plotly")
