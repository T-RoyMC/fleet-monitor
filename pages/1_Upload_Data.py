import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from common import (get_full_pipeline, engineer_features, build_feature_matrix)

st.set_page_config(page_title="Upload Data", page_icon="📂", layout="wide")
st.title("Upload Data")
st.caption("Upload a maintenance event log (CSV) in the same format as the Africa Synthetic Mining Equipment "
           "Failure dataset. The file is validated, summarised and scored with the model trained on the Model "
           "Training page (or a default class-weighted Logistic Regression if none has been trained).")

REQUIRED = ["equipment_id", "equipment_type", "manufacturer", "model_year", "mine_type", "country",
            "operating_hours", "event_date", "days_since_last_service",
            "oil_analysis_flag", "vibration_flag", "temperature_flag"]

up = st.file_uploader("Choose a CSV file", type=["csv"])

with st.expander("Required columns"):
    st.code(", ".join(REQUIRED))
    st.write("Optional: `event_type` (if present, `breakdown` is treated as the true failure label and the "
             "predictions are compared against it).")

if up is None:
    st.info("No file uploaded yet. Upload a CSV to validate and score it.")
    st.stop()

try:
    raw = pd.read_csv(up, parse_dates=["event_date"])
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

missing = [c for c in REQUIRED if c not in raw.columns]
if missing:
    st.error(f"The file is missing required columns: {', '.join(missing)}")
    st.stop()

st.success(f"File accepted: {up.name} — {len(raw):,} rows, {raw['equipment_id'].nunique():,} equipment units.")
st.session_state.uploaded_df = raw

a, b, c, d = st.columns(4)
a.metric("Events", f"{len(raw):,}")
b.metric("Equipment units", f"{raw['equipment_id'].nunique():,}")
c.metric("Date range", f"{raw['event_date'].min().date()} → {raw['event_date'].max().date()}")
d.metric("Missing values", f"{int(raw[REQUIRED].isna().sum().sum()):,}")

st.subheader("Preview")
st.dataframe(raw.head(15), use_container_width=True)

# ------------------------------------------------------------------ scoring
if "is_failure" not in raw.columns:
    raw["is_failure"] = (raw["event_type"] == "breakdown").astype(int) if "event_type" in raw.columns else 0
raw = raw.sort_values(["equipment_id", "event_date"]).reset_index(drop=True)
eng = engineer_features(raw)
feat = build_feature_matrix(eng)

model = st.session_state.get("trained_model")
name = st.session_state.get("trained_model_name")
if model is None:
    df_sorted, feat_sorted, split = get_full_pipeline()
    tr = feat_sorted.iloc[:split]
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(tr.drop(columns=["is_failure"]), tr["is_failure"])
    cols = list(tr.drop(columns=["is_failure"]).columns)
    name = "Logistic Regression (default)"
else:
    cols = list(st.session_state.trained_metrics["feature_names"])

X = feat.drop(columns=["is_failure"], errors="ignore").reindex(columns=cols, fill_value=0)
proba = model.predict_proba(X)[:, 1]
out = eng[["equipment_id", "equipment_type", "event_date"]].copy()
out["failure_probability"] = proba.round(3)
out["risk"] = pd.cut(proba, [-1, 0.4, 0.6, 2], labels=["Low", "Medium", "High"])

st.subheader(f"Failure-risk scores — {name}")
r1, r2, r3 = st.columns(3)
r1.metric("High-risk events", int((out["risk"] == "High").sum()))
r2.metric("Medium-risk events", int((out["risk"] == "Medium").sum()))
r3.metric("Low-risk events", int((out["risk"] == "Low").sum()))

col1, col2 = st.columns([3, 2])
with col1:
    st.dataframe(out.sort_values("failure_probability", ascending=False).head(25), use_container_width=True)
with col2:
    fig = go.Figure(go.Histogram(x=proba, nbinsx=25, marker_color="#4FA8D8"))
    fig.update_layout(title="Distribution of predicted failure probability", template="plotly_dark",
                      height=360, xaxis_title="Probability", yaxis_title="Events")
    st.plotly_chart(fig, use_container_width=True)

buf = io.StringIO()
out.to_csv(buf, index=False)
st.download_button("Download scored file (CSV)", buf.getvalue(), "scored_events.csv", "text/csv")
