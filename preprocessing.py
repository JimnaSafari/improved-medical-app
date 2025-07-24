"""
Data preprocessing functions for the AI Health Anomaly Detection app.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

@st.cache_data
def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Preprocess the health data by encoding categorical variables and scaling features.
    Returns the processed DataFrame, scaled DataFrame, and list of feature columns.
    """
    df = df.copy()
    df['activity_level'] = df['activity_level'].map({'low': 0, 'moderate': 1, 'high': 2})
    features = [
        'heart_rate', 'blood_oxygen', 'temperature', 'respiration_rate', 'activity_level',
        'systolic_bp', 'diastolic_bp', 'ecg', 'sleep_quality'
    ]
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    return df, df_scaled, features 