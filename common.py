"""Shared utilities for the Fleet Monitor Streamlit app.

Reuses the exact leakage-safe feature engineering pipeline from the
dissertation's EDA/modelling notebook, so results here are consistent
with what's reported in the write-up.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "train_data.csv"

LEAKY_COLS = ["event_type", "maintenance_type", "was_scheduled", "failure_mode",
              "component_failed", "severity", "downtime_hours", "repair_cost_usd",
              "production_loss_tonnes"]
ID_COLS = ["record_id", "equipment_id", "event_date", "prev_event_date"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["event_date"])
    df["is_failure"] = (df["event_type"] == "breakdown").astype(int)
    df = df.sort_values(["equipment_id", "event_date"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_year"] = df["event_date"].dt.year
    df["event_month"] = df["event_date"].dt.month
    df["equipment_age"] = df["event_year"] - df["model_year"]
    df["prior_event_count"] = df.groupby("equipment_id").cumcount()
    df["prior_failure_count"] = (
        df.groupby("equipment_id")["is_failure"]
          .apply(lambda s: s.shift(1).fillna(0).cumsum())
          .reset_index(level=0, drop=True)
    )
    df["prev_event_date"] = df.groupby("equipment_id")["event_date"].shift(1)
    df["days_since_prev_event"] = (df["event_date"] - df["prev_event_date"]).dt.days.fillna(-1)
    df["n_flags_active"] = df[["oil_analysis_flag", "vibration_flag", "temperature_flag"]].sum(axis=1)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in LEAKY_COLS + ID_COLS if c in df.columns]
    feat = df.drop(columns=drop_cols)
    feat = pd.get_dummies(feat, columns=["equipment_type", "manufacturer", "mine_type", "country"])
    bool_cols = feat.select_dtypes(include="bool").columns
    feat[bool_cols] = feat[bool_cols].astype(int)
    return feat


def time_split(df: pd.DataFrame, feat: pd.DataFrame, frac: float = 0.8):
    """Returns (df_sorted, feat_sorted, split_index) using a time-based split."""
    sort_idx = df.sort_values("event_date").index
    feat_sorted = feat.loc[sort_idx].reset_index(drop=True)
    df_sorted = df.loc[sort_idx].reset_index(drop=True)
    split = int(len(feat_sorted) * frac)
    return df_sorted, feat_sorted, split


def get_full_pipeline():
    """One-call helper: raw -> engineered -> feature matrix -> time split."""
    raw = load_raw()
    eng = engineer_features(raw)
    feat = build_feature_matrix(eng)
    df_sorted, feat_sorted, split = time_split(eng, feat)
    return df_sorted, feat_sorted, split


def top_units(df: pd.DataFrame, n: int = 8):
    counts = df.groupby("equipment_id").size().sort_values(ascending=False)
    return counts.head(n).index.tolist()


def unit_state(row: pd.Series) -> str:
    """Map a real dataset row to Normal / Fault / Failure, consistent with
    the EDA finding that n_flags_active is the strongest predictor."""
    if row["is_failure"] == 1:
        return "failure"
    if row["n_flags_active"] >= 1:
        return "fault"
    return "ok"
