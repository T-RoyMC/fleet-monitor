import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from common import load_raw, engineer_features

st.set_page_config(page_title="Live Visualizations", page_icon="📈", layout="wide")
st.title("Dataset Replay")
st.caption("Steps through the dataset in chronological order — the charts below grow as each batch of events is \"read,\" simulating a live feed.")

@st.cache_data
def get_data():
    df = engineer_features(load_raw())
    return df.sort_values("event_date").reset_index(drop=True)

df = get_data()
N = len(df)

if "row_ptr" not in st.session_state:
    st.session_state.row_ptr = 0
if "replay_autoplay" not in st.session_state:
    st.session_state.replay_autoplay = False
if "roll_dates" not in st.session_state:
    st.session_state.roll_dates = []
    st.session_state.roll_failure_rate = []
    st.session_state.roll_cum_events = []
    st.session_state.roll_flag_rate = []

BATCH = 50

def consume_batch():
    start = st.session_state.row_ptr
    end = min(start + BATCH, N)
    if start >= N:
        return False
    chunk = df.iloc[:end]  # cumulative view up to `end`
    st.session_state.roll_dates.append(chunk["event_date"].iloc[-1])
    st.session_state.roll_failure_rate.append(chunk["is_failure"].mean())
    st.session_state.roll_cum_events.append(end)
    st.session_state.roll_flag_rate.append((chunk["n_flags_active"] > 0).mean())
    st.session_state.row_ptr = end
    return True

c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
with c1:
    if st.button("⏭ Read next batch"):
        consume_batch()
with c2:
    st.session_state.replay_autoplay = st.checkbox("▶ Auto-play", value=st.session_state.replay_autoplay)
with c3:
    if st.button("↺ Reset"):
        st.session_state.row_ptr = 0
        st.session_state.roll_dates = []
        st.session_state.roll_failure_rate = []
        st.session_state.roll_cum_events = []
        st.session_state.roll_flag_rate = []
        st.rerun()
with c4:
    st.progress(min(st.session_state.row_ptr / N, 1.0), text=f"{st.session_state.row_ptr:,} / {N:,} events read")

col1, col2 = st.columns(2)
with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=st.session_state.roll_cum_events, y=st.session_state.roll_failure_rate,
                               mode="lines+markers", line=dict(color="#E24B4B")))
    fig1.update_layout(title="Cumulative failure rate as events accumulate", template="plotly_dark",
                        xaxis_title="Events read", yaxis_title="Failure rate", height=380,
                        yaxis_range=[0, 1])
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=st.session_state.roll_cum_events, y=st.session_state.roll_flag_rate,
                               mode="lines+markers", line=dict(color="#E8A93C")))
    fig2.update_layout(title="Share of events with an active condition-monitoring flag", template="plotly_dark",
                        xaxis_title="Events read", yaxis_title="Flag-active rate", height=380,
                        yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

if st.session_state.row_ptr > 0:
    seen = df.iloc[:st.session_state.row_ptr]
    st.subheader("Snapshot of data read so far")
    a, b, c, d = st.columns(4)
    a.metric("Events read", f"{len(seen):,}")
    b.metric("Failures seen", int(seen["is_failure"].sum()))
    c.metric("Unique equipment", seen["equipment_id"].nunique())
    d.metric("Date range", f"{seen['event_date'].min().date()} → {seen['event_date'].max().date()}")

    fig3 = go.Figure()
    counts = seen["equipment_type"].value_counts()
    fig3.add_trace(go.Bar(x=counts.index, y=counts.values, marker_color="#4FA8D8"))
    fig3.update_layout(title="Equipment type distribution (data read so far)", template="plotly_dark", height=350)
    st.plotly_chart(fig3, use_container_width=True)

if st.session_state.replay_autoplay:
    more = consume_batch()
    if not more:
        st.session_state.replay_autoplay = False
    time.sleep(0.5)
    st.rerun()
