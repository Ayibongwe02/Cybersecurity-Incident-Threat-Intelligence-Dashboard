# CyberSentinel — Threat Intelligence Dashboard (Streamlit)

A Streamlit rebuild of the original vanilla-JS CyberSentinel dashboard, with
**live** volumetric forecasting and behavioral anomaly detection instead of
a pre-generated JSON snapshot.

## What changed from the original

| | Original | This version |
|---|---|---|
| Forecasting | Prophet (heavy install, C++ Stan compile step) | `statsmodels` Holt-Winters / ARIMA — same forecasting quality, far lighter and reliable to deploy |
| Anomaly detection | scikit-learn Isolation Forest, pre-computed into JSON | Same Isolation Forest, computed **live** in-app with an adjustable sensitivity slider |
| Frontend | Vanilla JS + Chart.js, polls a static JSON every 60s | Streamlit + Plotly, recomputes on interaction |
| Data input | Fixed CSVs baked into the pipeline | Same bundled CSVs, plus a sidebar CSV uploader for your own data |

Prophet was dropped specifically because it's a common cause of Streamlit
Cloud build failures/timeouts (large dependency tree, native compilation).
`statsmodels` produces comparable forecasts without that risk.

## Pages

- **Overview** — Security Posture Index, blocked payload volume, active mitigations, anomaly rate, incident volume by severity, traffic by host
- **Endpoints** — per-host risk score, traffic/latency history, raw event log
- **Predictive Engine** — live Holt-Winters/ARIMA forecast with backtest RMSE/MAPE and a 95% confidence band
- **Threat Hunting** — live Isolation Forest anomaly log with classification filters and risk scores

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or with Docker:

```bash
docker compose up --build
```

Open **http://localhost:8501**.

## Using your own data

Upload in the sidebar:
- **Weekly traffic CSV:** `ds` (date), `y` (numeric volume)
- **Incidents CSV:** `timestamp, host, payload_size_kb, response_latency_ms, failed_auth_count, bytes_out_mb, request_rate, severity, event_type, source_ip, destination_port`

If a required column is missing, the app shows an error and falls back to
the bundled sample data.

## Troubleshooting (Streamlit Community Cloud)

**Dependency install fails ("installer returned a non-zero exit code"):**
`requirements.txt` uses minimum-version ranges, not exact pins, so pip can
resolve to whatever has a prebuilt package for the Python version Cloud
uses. Don't rely on `runtime.txt` to force a Python version — it's
currently unreliable on Cloud; set it explicitly in "Advanced settings"
at deploy time instead.

**`FileNotFoundError` reading a CSV:** the app checks both `data/<file>.csv`
and `<file>.csv` at the repo root, so it works whether you `git push`
(preserves folders) or use GitHub's one-by-one "Add files via upload"
(flattens everything to root). If you still hit this, run `git ls-files data/`
to confirm the files are actually committed.

## Tech stack

- **App/UI:** Streamlit, Plotly
- **Forecasting:** statsmodels (Holt-Winters, ARIMA)
- **Anomaly detection:** scikit-learn (Isolation Forest)
- **Deployment:** Docker / docker-compose, or plain Python, or Streamlit Community Cloud
