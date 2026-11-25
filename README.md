# Underground Fleet Monitor — Streamlit App

A 4-page predictive maintenance dashboard driven by the real
`train_data.csv` dataset (Africa Synthetic Mining Equipment Failure,
Electric Sheep Africa, 2026) and the same leakage-safe feature
engineering pipeline used throughout the dissertation.

## Pages
1. **Home — Fleet Monitor**: 8 real equipment units (the ones with the
   most recorded history), stepping through their actual event
   sequences with Normal/Fault/Failure states, an alert log, and audio
   alerts (Web Audio API — click anywhere on the page once first, or
   most browsers block autoplay audio until a user gesture).
2. **Live Visualizations**: replays the full dataset in chronological
   order in batches of 50 events, growing rolling-failure-rate and
   flag-activity charts live.
3. **Model Training**: pick Logistic Regression / Random Forest /
   Gradient Boosting, click Train, and it refits once on the real
   time-based 80/20 split with a progress bar, then shows accuracy /
   F1 / ROC-AUC, a ROC curve, a confusion matrix, and feature
   importances.
4. **Live Predictions**: streams the held-out test set through
   whichever model you trained on page 3, one event at a time, showing
   predicted probability vs. actual outcome and a running accuracy/F1.

## Run it

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Then open the local URL it prints (usually http://localhost:8501).

## Notes

- `common.py` holds the shared data loading, feature engineering, and
  time-split logic — identical to the pipeline described in Chapter 3
  of the dissertation.
- Auto-play on pages 1, 2 and 4 uses a `time.sleep()` + `st.rerun()`
  loop, the standard Streamlit pattern for live-updating pages.
- Page 4 requires training a model on page 3 first in the same
  session (models aren't saved to disk between runs).
