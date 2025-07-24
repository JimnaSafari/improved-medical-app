"""
Anomaly detection and evaluation functions for the AI Health Anomaly Detection app.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

def detect_anomalies(df_scaled: pd.DataFrame, contamination: float = 0.05) -> tuple[np.ndarray, IsolationForest]:
    """
    Detect anomalies in the scaled data using Isolation Forest.
    Returns predictions and the trained model.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(df_scaled)
    return preds, model

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Evaluate the anomaly detection model using classification metrics.
    Returns a DataFrame with the evaluation report.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    return pd.DataFrame(report).transpose() 