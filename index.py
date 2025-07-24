# Streamlit AI Health Anomaly Detection System (Large-Scale Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta

from data_simulation import simulate_health_data
from preprocessing import preprocess_data
from anomaly_detection import detect_anomalies, evaluate_model

# -----------------------------
# SECTION 5: Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Health Anomaly Detection", layout="wide")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("Upload your own health data or use simulated data.")
# File uploader for user CSV
uploaded_file = st.sidebar.file_uploader("Upload your health data (CSV)", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Simulation Settings**")
st.sidebar.caption("If no CSV is uploaded, simulated data will be used.")
num_users = st.sidebar.slider("Number of Users", 1, 10, 3, help="Number of users to simulate.")
num_minutes = st.sidebar.slider("Minutes of Data per User", 100, 1000, 300, help="How many minutes of data per user.")
contamination = st.sidebar.slider("Anomaly Rate (contamination)", 0.01, 0.2, 0.05, help="Proportion of anomalies expected in the data.")
st.sidebar.markdown("---")

# Main UI
st.title("🧠 AI-Powered Health Anomaly Detection")
st.markdown("---")

# Data simulation or upload and display
st.header("📊 Health Data")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = {
            'user_id', 'timestamp', 'heart_rate', 'blood_oxygen', 'temperature', 'respiration_rate', 'activity_level',
            'systolic_bp', 'diastolic_bp', 'ecg', 'sleep_quality'
        }
        if not required_cols.issubset(df.columns):
            st.warning(f"CSV is missing required columns: {required_cols - set(df.columns)}. Using simulated data instead.")
            with st.spinner("Simulating health data..."):
                df = simulate_health_data(num_users, num_minutes)
        else:
            st.success("✅ CSV uploaded and loaded successfully!")
    except Exception as e:
        st.warning(f"Failed to read CSV: {e}. Using simulated data instead.")
        with st.spinner("Simulating health data..."):
            df = simulate_health_data(num_users, num_minutes)
else:
    with st.spinner("Simulating health data..."):
        df = simulate_health_data(num_users, num_minutes)  # Generate synthetic health data
df_display = df.head(100)  # Show only first 100 rows for performance

# Summary card
st.markdown("---")
st.subheader("📋 Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Users", df['user_id'].nunique())
col2.metric("Records", len(df))
col3.metric("Time Range", f"{df['timestamp'].min()} to {df['timestamp'].max()}")
st.markdown("---")

with st.expander("Show raw data (first 100 rows)"):
    st.dataframe(df_display)

# Preprocessing and anomaly detection
st.header("🧪 Anomaly Detection")
with st.spinner("Preprocessing data and detecting anomalies..."):
    df_processed, df_scaled, feature_cols = preprocess_data(df)  # Encode and scale features
    preds, model = detect_anomalies(df_scaled, contamination)  # Run anomaly detection
    # Label each row as 'Anomaly' or 'Normal' based on model prediction
    df_processed['anomaly'] = ['Anomaly' if x == -1 else 'Normal' for x in preds]
st.markdown("---")
with st.expander("Show anomaly detection results (first 100 rows)"):
    st.dataframe(df_processed[[
        'user_id', 'timestamp', 'heart_rate', 'blood_oxygen', 'temperature',
        'systolic_bp', 'diastolic_bp', 'ecg', 'sleep_quality',
        'anomaly'
    ]].head(100))

# Evaluation setup
st.header("📈 Model Evaluation")
with st.spinner("Evaluating model..."):
    df_processed['anomaly_label'] = df_processed['anomaly'].map({'Normal': 0, 'Anomaly': 1})  # Convert labels to 0/1
    X = df_scaled
    y = df_processed['anomaly_label']
    # Split data for evaluation (note: unsupervised, so labels are synthetic)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert model output to 0/1
    except Exception as e:
        st.error(f"Error fitting or predicting model: {e}")
        y_pred = np.zeros(len(y_test)) # Return all zeros as a fallback

# Display evaluation report
eval_df = evaluate_model(y_test, y_pred)
st.dataframe(eval_df)

# Visualize anomalies
st.header("📉 Anomaly Visualization")
fig, ax = plt.subplots()
anomaly_points = df_processed[df_processed['anomaly'] == 'Anomaly']  # Filter anomaly rows
# Plot heart rate over time for each user, highlight anomalies
sns.lineplot(data=df_processed, x='timestamp', y='heart_rate', hue='user_id', ax=ax)
plt.scatter(anomaly_points['timestamp'], anomaly_points['heart_rate'], color='red', label='Anomaly')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

# Interactive Plotly visualization
st.header("🖱️ Interactive Anomaly Exploration")
metric_options = [
    'heart_rate', 'blood_oxygen', 'temperature', 'respiration_rate',
    'systolic_bp', 'diastolic_bp', 'ecg', 'sleep_quality'
]
selected_metric = st.selectbox("Select metric to visualize:", metric_options, index=0)

fig_plotly = px.line(
    df_processed,
    x='timestamp',
    y=selected_metric,
    color='user_id',
    title=f"{selected_metric.replace('_', ' ').title()} Over Time by User",
    labels={'timestamp': 'Timestamp', selected_metric: selected_metric.replace('_', ' ').title()}
)
# Overlay anomalies as red markers
anomaly_df = df_processed[df_processed['anomaly'] == 'Anomaly']
fig_plotly.add_scatter(
    x=anomaly_df['timestamp'],
    y=anomaly_df[selected_metric],
    mode='markers',
    marker=dict(color='red', size=8, symbol='x'),
    name='Anomaly',
    showlegend=True
)
st.plotly_chart(fig_plotly, use_container_width=True)

# Advanced analytics section
st.markdown("---")
with st.expander("📊 Advanced Analytics", expanded=False):
    st.subheader("Anomaly Counts per User")
    anomaly_counts = df_processed.groupby('user_id')['anomaly'].value_counts().unstack(fill_value=0)
    st.dataframe(anomaly_counts)

    st.subheader("Feature Importance (Isolation Forest)")
    try:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
    except Exception as e:
        st.info("Feature importances not available for this model.")

    st.subheader("Correlation Heatmap (All Metrics)")
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = df_processed[feature_cols].corr()
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# Optional save
st.sidebar.markdown("---")
# Allow user to export anomaly report as CSV using a download button
try:
    csv = df_processed.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="💾 Download Anomaly Report",
        data=csv,
        file_name="anomaly_report.csv",
        mime="text/csv"
    )
except Exception as e:
    st.sidebar.error(f"Failed to generate download: {e}")
