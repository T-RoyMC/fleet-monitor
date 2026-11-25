import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from common import get_full_pipeline

st.set_page_config(page_title="Model Training", page_icon="🛠️", layout="wide")
st.title("Model Training")
st.caption("Trains on a time-based 80/20 split of the dataset — the same leakage-safe features and split used throughout the dissertation. Click Train to refit.")

MODEL_CHOICES = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=6, class_weight="balanced", random_state=42),
}

choice = st.selectbox("Model to train", list(MODEL_CHOICES.keys()))

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.trained_model_name = None
    st.session_state.trained_metrics = None

if st.button("▶ Train", type="primary"):
    progress = st.progress(0, text="Loading data...")
    df_sorted, feat_sorted, split = get_full_pipeline()
    progress.progress(20, text="Preparing leakage-safe feature matrix...")
    time.sleep(0.3)

    train, test = feat_sorted.iloc[:split], feat_sorted.iloc[split:]
    X_train, y_train = train.drop(columns=["is_failure"]), train["is_failure"]
    X_test, y_test = test.drop(columns=["is_failure"]), test["is_failure"]
    progress.progress(45, text="Applying time-based train/test split (80/20)...")
    time.sleep(0.3)

    model = MODEL_CHOICES[choice]()
    progress.progress(65, text=f"Fitting {choice}...")
    if choice == "Logistic Regression":
        model.fit(X_train, y_train)
    else:
        sw = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=sw)
    time.sleep(0.3)

    progress.progress(90, text="Evaluating on held-out test set...")
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "y_test": y_test, "proba": proba, "pred": pred,
        "feature_names": X_train.columns,
    }
    if hasattr(model, "feature_importances_"):
        metrics["importances"] = pd.Series(model.feature_importances_, index=X_train.columns)

    st.session_state.trained_model = model
    st.session_state.trained_model_name = choice
    st.session_state.trained_metrics = metrics
    progress.progress(100, text="Done.")
    time.sleep(0.2)
    progress.empty()
    st.success(f"{choice} trained and evaluated on the held-out (most recent 20%) test set.")

if st.session_state.trained_metrics is not None:
    m = st.session_state.trained_metrics
    st.subheader(f"Results — {st.session_state.trained_model_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("F1 (failure class)", f"{m['f1']:.3f}")
    c3.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        fpr, tpr, _ = roc_curve(m["y_test"], m["proba"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#4FA8D8"), name="ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#5B6B73", dash="dash"), name="Chance"))
        fig.update_layout(title="ROC curve", template="plotly_dark", height=380,
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cm = confusion_matrix(m["y_test"], m["pred"])
        fig = go.Figure(data=go.Heatmap(z=cm, x=["No Failure", "Failure"], y=["No Failure", "Failure"],
                                         colorscale="Blues", showscale=False, text=cm, texttemplate="%{text}"))
        fig.update_layout(title="Confusion matrix", template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    if "importances" in m:
        top = m["importances"].sort_values(ascending=False).head(12)
        fig = go.Figure(go.Bar(x=top.values[::-1], y=top.index[::-1], orientation="h", marker_color="#8172B2"))
        fig.update_layout(title="Top feature importances", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.info("This trained model is now available on the **Live Predictions** page.")
else:
    st.info("No model trained yet this session — select a model above and click Train.")
