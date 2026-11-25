import time
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from common import load_raw, engineer_features, top_units, unit_state

st.set_page_config(page_title="Fleet Monitor", page_icon="⚠️", layout="wide")

# ---------------------------------------------------------------- data ----
@st.cache_data
def get_data():
    return engineer_features(load_raw())

df = get_data()
UNITS = top_units(df, n=8)
histories = {u: df[df["equipment_id"] == u].reset_index(drop=True) for u in UNITS}

# ---------------------------------------------------------- session state
if "ptrs" not in st.session_state:
    st.session_state.ptrs = {u: 0 for u in UNITS}
if "prev_states" not in st.session_state:
    st.session_state.prev_states = {u: "ok" for u in UNITS}
if "sound" not in st.session_state:
    st.session_state.sound = True
if "log" not in st.session_state:
    st.session_state.log = []
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False

EQUIP_LABEL = {
    "haul_truck": "Haul Truck", "loader": "LHD Loader", "drill": "Drill Rig",
    "crusher": "Crusher Unit", "pump": "Dewatering Pump", "excavator": "Excavator",
    "conveyor": "Belt Conveyor",
}

def advance_one():
    beeps = []
    for u in UNITS:
        hist = histories[u]
        ptr = st.session_state.ptrs[u]
        row = hist.iloc[ptr]
        state = unit_state(row)
        prev = st.session_state.prev_states[u]
        if state != prev:
            level = "resolve" if state == "ok" else ("warn" if state == "fault" else "crit")
            ts = row["event_date"].strftime("%Y-%m-%d")
            msg = f"{EQUIP_LABEL.get(row['equipment_type'], row['equipment_type'])} ({u}) — event on {ts}: {'failure — maintenance required' if state=='failure' else ('fault — flags active' if state=='fault' else 'returned to normal')}"
            st.session_state.log.insert(0, {"level": level, "msg": msg})
            if state == "fault":
                beeps.append("fault")
            elif state == "failure":
                beeps.append("failure")
        st.session_state.prev_states[u] = state
        st.session_state.ptrs[u] = (ptr + 1) % len(hist)
    st.session_state.log = st.session_state.log[:60]
    return beeps

# ------------------------------------------------------------- controls --
st.title("Underground Fleet Monitor")
st.caption("Driven by real event history from the Africa Synthetic Mining Equipment Failure dataset — stepping through each unit's actual recorded events, not a random simulation.")

c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
with c1:
    if st.button("⏭ Advance one step"):
        advance_one()
with c2:
    st.session_state.autoplay = st.checkbox("▶ Auto-play", value=st.session_state.autoplay)
with c3:
    st.session_state.sound = st.checkbox("🔊 Sound", value=st.session_state.sound)
with c4:
    if st.button("↺ Reset"):
        st.session_state.ptrs = {u: 0 for u in UNITS}
        st.session_state.prev_states = {u: "ok" for u in UNITS}
        st.session_state.log = []
        st.rerun()

# --------------------------------------------------------- fleet summary
counts = {"ok": 0, "fault": 0, "failure": 0}
for u in UNITS:
    counts[st.session_state.prev_states[u]] += 1

m1, m2, m3, m4 = st.columns(4)
m1.metric("Normal operation", counts["ok"])
m2.metric("Fault — emerging risk", counts["fault"])
m3.metric("Failure — action required", counts["failure"])
m4.metric("Units monitored", len(UNITS))

STATE_COLOR = {"ok": "#33C481", "fault": "#E8A93C", "failure": "#E24B4B"}
STATE_LABEL = {"ok": "NORMAL", "fault": "FAULT", "failure": "FAILURE"}

# ------------------------------------------------------------- fleet grid
grid_cols = st.columns(4)
for i, u in enumerate(UNITS):
    hist = histories[u]
    ptr = st.session_state.ptrs[u]
    row = hist.iloc[ptr]
    state = st.session_state.prev_states[u]
    color = STATE_COLOR[state]
    with grid_cols[i % 4]:
        st.markdown(f"""
<div style="border:1px solid {color}55; border-left:4px solid {color}; border-radius:8px;
            padding:12px; margin-bottom:12px; background:#12171B;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <div style="font-weight:600; color:#E7EDF0; font-size:14px;">{EQUIP_LABEL.get(row['equipment_type'], row['equipment_type'])}</div>
      <div style="font-family:monospace; font-size:11px; color:#5B6B73;">{u}</div>
    </div>
    <div style="font-family:monospace; font-size:10px; font-weight:700; color:{color};
                background:{color}22; padding:3px 8px; border-radius:3px;">{STATE_LABEL[state]}</div>
  </div>
  <div style="margin-top:8px; font-size:12px; color:#8FA0A8; line-height:1.6;">
    Operating hours: <b style="color:#E7EDF0;">{row['operating_hours']:.0f}</b><br>
    Days since service: <b style="color:#E7EDF0;">{row['days_since_last_service']:.0f}</b><br>
    Event: <span style="color:#E7EDF0;">{row['event_type']}</span> ({row['severity']})
  </div>
  <div style="display:flex; gap:6px; margin-top:8px;">
    <div style="flex:1; text-align:center; font-size:10px; font-family:monospace; padding:4px;
                border-radius:4px; background:{'#E8A93C22' if row['oil_analysis_flag'] else '#161D22'};
                color:{'#E8A93C' if row['oil_analysis_flag'] else '#5B6B73'};">OIL</div>
    <div style="flex:1; text-align:center; font-size:10px; font-family:monospace; padding:4px;
                border-radius:4px; background:{'#E8A93C22' if row['vibration_flag'] else '#161D22'};
                color:{'#E8A93C' if row['vibration_flag'] else '#5B6B73'};">VIB</div>
    <div style="flex:1; text-align:center; font-size:10px; font-family:monospace; padding:4px;
                border-radius:4px; background:{'#E8A93C22' if row['temperature_flag'] else '#161D22'};
                color:{'#E8A93C' if row['temperature_flag'] else '#5B6B73'};">TEMP</div>
  </div>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- log ----
st.subheader("Alert log")
log_html = "".join(
    f"<div style='padding:6px 10px; margin-bottom:6px; border-left:3px solid "
    f"{'#33C481' if e['level']=='resolve' else ('#E8A93C' if e['level']=='warn' else '#E24B4B')}; "
    f"background:#161D22; border-radius:4px; font-size:12px; color:#E7EDF0;'>{e['msg']}</div>"
    for e in st.session_state.log
) or "<div style='color:#5B6B73; font-size:12px;'>No events yet — click Advance or Auto-play.</div>"
st.markdown(f"<div style='max-height:260px; overflow-y:auto;'>{log_html}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------- sound ----
if st.session_state.sound:
    components.html("""
    <script>
    function beep(freq, dur, type) {
      try {
        const ctx = new (window.AudioContext||window.webkitAudioContext)();
        const osc = ctx.createOscillator(); const gain = ctx.createGain();
        osc.type = type; osc.frequency.value = freq; gain.gain.value = 0.05;
        osc.connect(gain); gain.connect(ctx.destination); osc.start();
        gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur);
        osc.stop(ctx.currentTime + dur);
      } catch(e) {}
    }
    window.__pmBeep = beep;
    </script>
    """, height=0)

# ------------------------------------------------------------ autoplay --
if st.session_state.autoplay:
    for _ in range(1):
        beeps = advance_one()
        if st.session_state.sound and beeps:
            tone = "failure" if "failure" in beeps else "fault"
            freq = 880 if tone == "failure" else 660
            components.html(f"<script>if(window.parent.document.querySelector('iframe')){{}}</script>"
                             f"<script>try{{const ctx=new (window.AudioContext||window.webkitAudioContext)();"
                             f"const o=ctx.createOscillator();const g=ctx.createGain();o.type='sine';"
                             f"o.frequency.value={freq};g.gain.value=0.05;o.connect(g);g.connect(ctx.destination);"
                             f"o.start();g.gain.exponentialRampToValueAtTime(0.0001,ctx.currentTime+0.15);"
                             f"o.stop(ctx.currentTime+0.15);}}catch(e){{}}</script>", height=0)
    time.sleep(1.1)
    st.rerun()

st.caption("Real dataset, prototype alert logic — not a production monitoring system.")
