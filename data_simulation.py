"""
Data simulation functions for the AI Health Anomaly Detection app.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data
def simulate_health_data(num_users: int = 5, minutes: int = 500) -> pd.DataFrame:
    """
    Simulate health data for a given number of users and minutes.
    Returns a DataFrame with synthetic health metrics.
    """
    user_ids = [f'user_{i+1}' for i in range(num_users)]
    start_time = datetime.now()
    data = []

    for user in user_ids:
        timestamp = start_time
        for _ in range(minutes):
            data.append({
                'user_id': user,
                'timestamp': timestamp,
                'heart_rate': np.random.randint(60, 100),
                'blood_oxygen': np.random.randint(90, 100),
                'temperature': np.random.normal(36.5, 0.5),
                'respiration_rate': np.random.randint(12, 20),
                'activity_level': np.random.choice(['low', 'moderate', 'high']),
                'systolic_bp': np.random.randint(110, 140),
                'diastolic_bp': np.random.randint(70, 90),
                'ecg': np.random.normal(0, 1),
                'sleep_quality': np.random.randint(1, 6)
            })
            timestamp += timedelta(minutes=1)

    df = pd.DataFrame(data)
    return df 