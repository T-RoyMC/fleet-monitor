import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score
from common import get_full_pipeline

st.set_page_config(page_title="Live Predictions", page_icon="🔮", layout="wide")
st.title("Live Predictions")
st.caption("Streams rows from the held-out test set through the model trained on the Model Training page, one at a time.")

if st.session_state.get("trained_model") is None:
    st.warning("No trained model in this session yet. Go to **Model Training**, pick a model, and click Train first.")
    st.stop()

model = st.session_state.trained_model
model_name = st.session_state.trained_model_name

@st.cache_data
def get_test_set():
    df_sorted, feat_sorted, split = get_full_pipeline()
    test_df = df_sorted.iloc[split:].reset_index(drop=True)
    test_feat = feat_sorted.iloc[split:].reset_index(drop=True)
    return test_df, test_feat

test_df, test_feat = get_test_set()
X_test = test_feat.drop(columns=["is_failure"])
N = len(X_test)

st.caption(f"Model in use: **{model_name}** · Test stream: {N} held-out events, in chronological order")

if "pred_ptr" not in st.session_state:
    st.session_state.pred_ptr = 0
    st.session_state.pred_log = []
    st.session_state.preds = []
    st.session_state.actuals = []
if "pred_autoplay" not in st.session_state:
    st.session_state.pred_autoplay = False

c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
with c1:
    if st.button("⏭ Next event"):
        step = True
    else:
        step = False
with c2:
    st.session_state.pred_autoplay = st.checkbox("▶ Auto-play", value=st.session_state.pred_autoplay)
with c3:
    if st.button("↺ Reset stream"):
        st.session_state.pred_ptr = 0
        st.session_state.pred_log = []
        st.session_state.preds = []
        st.session_state.actuals = []
        st.rerun()
with c4:
    st.progress(min(st.session_state.pred_ptr / N, 1.0) if N else 0,
                text=f"{st.session_state.pred_ptr} / {N} events streamed")

def advance_prediction():
    i = st.session_state.pred_ptr
    if i >= N:
        return False
    row_feat = X_test.iloc[[i]]
    row_raw = test_df.iloc[i]
    proba = model.predict_proba(row_feat)[0, 1]
    pred = int(proba >= 0.5)
    actual = int(row_raw["is_failure"])
    correct = pred == actual
    st.session_state.preds.append(pred)
    st.session_state.actuals.append(actual)
    st.session_state.pred_log.insert(0, {
        "date": row_raw["event_date"].strftime("%Y-%m-%d"),
        "equipment": row_raw["equipment_id"],
        "proba": proba, "pred": pred, "actual": actual, "correct": correct,
    })
    st.session_state.pred_log = st.session_state.pred_log[:80]
    st.session_state.pred_ptr = i + 1
    return True

if step:
    advance_prediction()

if st.session_state.preds:
    a, b, c = st.columns(3)
    a.metric("Events predicted", len(st.session_state.preds))
    a_acc = accuracy_score(st.session_state.actuals, st.session_state.preds)
    b.metric("Running accuracy", f"{a_acc:.3f}")
    try:
        a_f1 = f1_score(st.session_state.actuals, st.session_state.preds)
    except Exception:
        a_f1 = 0.0
    c.metric("Running F1 (failure)", f"{a_f1:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[p["proba"] for p in reversed(st.session_state.pred_log[-100:])],
                              mode="lines+markers", line=dict(color="#4FA8D8"), name="Predicted probability"))
    fig.add_trace(go.Scatter(y=[p["actual"] for p in reversed(st.session_state.pred_log[-100:])],
                              mode="markers", marker=dict(color="#E24B4B", size=7), name="Actual (0/1)"))
    fig.update_layout(title="Predicted failure probability vs. actual outcome (most recent events)",
                       template="plotly_dark", height=380, yaxis_range=[-0.05, 1.05])
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction log")
if st.session_state.pred_log:
    rows = []
    for p in st.session_state.pred_log:
        rows.append({
            "Date": p["date"], "Equipment": p["equipment"],
            "P(failure)": f"{p['proba']:.3f}",
            "Predicted": "Failure" if p["pred"] else "No failure",
            "Actual": "Failure" if p["actual"] else "No failure",
            "Correct": "✓" if p["correct"] else "✗",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)
else:
    st.info("No predictions streamed yet — click Next event or Auto-play.")

if st.session_state.pred_autoplay:
    more = advance_prediction()
    if not more:
        st.session_state.pred_autoplay = False
    time.sleep(0.4)
    st.rerun()
