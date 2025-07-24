# AI Health Anomaly Detection Dashboard

A Streamlit-powered web app for simulating, uploading, and analyzing health data using AI-based anomaly detection. Visualize anomalies, explore advanced analytics, and export reports—all in an interactive dashboard.

---

## 🚀 Features
- **Simulate or Upload Data:** Use built-in health data simulation or upload your own CSV.
- **Rich Health Metrics:** Analyze heart rate, blood oxygen, temperature, respiration rate, activity level, blood pressure, ECG, and sleep quality.
- **AI Anomaly Detection:** Detect anomalies using Isolation Forest.
- **Interactive Visualizations:** Explore data and anomalies with both static and interactive (Plotly) plots.
- **Advanced Analytics:**
  - Anomaly counts per user
  - Feature importance (if available)
  - Correlation heatmap for all metrics
- **Export:** Download anomaly reports as CSV.

---

## 🛠️ Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Medical_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run index.py
   ```

---

## 📊 Usage

- **Upload Data:** Use the sidebar to upload your own CSV file (see required columns below), or use the simulation controls to generate synthetic data.
- **Configure Simulation:** Adjust the number of users, minutes per user, and expected anomaly rate.
- **Explore Data:**
  - View summary stats and raw data.
  - See anomaly detection results and download the report.
  - Use interactive plots to explore any metric and highlight anomalies.
  - Open the "Advanced Analytics" expander for anomaly counts, feature importance, and a correlation heatmap.

---

## 📁 CSV Format
If uploading your own data, your CSV must include these columns:
- `user_id`, `timestamp`, `heart_rate`, `blood_oxygen`, `temperature`, `respiration_rate`, `activity_level`, `systolic_bp`, `diastolic_bp`, `ecg`, `sleep_quality`

Example:
```csv
user_id,timestamp,heart_rate,blood_oxygen,temperature,respiration_rate,activity_level,systolic_bp,diastolic_bp,ecg,sleep_quality
user_1,2024-01-01 00:00:00,72,98,36.7,16,moderate,120,80,0.1,4
...
```

---

## ⚙️ Customization
- **Add More Metrics:** Edit `simulate_health_data` and `preprocess_data` in `index.py`.
- **Change Model:** Swap out Isolation Forest for another model in `detect_anomalies`.
- **UI Tweaks:** Adjust Streamlit layout, colors, or add new visualizations as needed.

---

## 📄 License
MIT License (or specify your own)

---

## 🙏 Acknowledgments
- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), [Plotly](https://plotly.com/python/), [Seaborn](https://seaborn.pydata.org/), and [Pandas](https://pandas.pydata.org/). 