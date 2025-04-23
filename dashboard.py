import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from collections import defaultdict

# Streamlit page config
st.set_page_config(page_title="Bike Activity Dashboard – Konstanz (March 2025)", layout="wide")

# Title
st.title("🚴‍♂️ Bike Activity Dashboard – Konstanz (March 2025)")

# Load data
@st.cache_data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["bike_id", "timestamp"])
    return df

df = load_data("konstanz_march_2025.jsonl")

# Section 1: Rentals Logic (Hourly unique rentals by hour, weekday vs weekend)
rentals = []

for bike_id, group in df.groupby("bike_id"):
    group = group.sort_values("timestamp")
    group = group.reset_index(drop=True)
    for i in range(1, len(group)):
        prev_row = group.iloc[i - 1]
        curr_row = group.iloc[i]
        if (prev_row["lat"] != curr_row["lat"]) or (prev_row["lng"] != curr_row["lng"]):
            rentals.append(curr_row)

df_rentals = pd.DataFrame(rentals)
df_rentals["hour"] = df_rentals["timestamp"].dt.hour
df_rentals["day_of_week"] = df_rentals["timestamp"].dt.weekday
df_rentals["day_type"] = df_rentals["day_of_week"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")

hourly_rentals = df_rentals.groupby(["hour", "day_type"])["bike_id"].nunique().reset_index()

# Section 2: Station Idle Time Logic
bike_events = df.sort_values(["bike_id", "timestamp"])
bike_events["next_timestamp"] = bike_events.groupby("bike_id")["timestamp"].shift(-1)
bike_events["next_place"] = bike_events.groupby("bike_id")["place_name"].shift(-1)

bike_events["idle_duration"] = (bike_events["next_timestamp"] - bike_events["timestamp"]).dt.total_seconds() / 3600
bike_events = bike_events[bike_events["place_name"] == bike_events["next_place"]]

idle_by_station = bike_events.groupby("place_name")["idle_duration"].mean().sort_values(ascending=False).head(10)

# Section 3: Idle Duration Summary
all_idle_durations = bike_events["idle_duration"].dropna()
avg_idle = all_idle_durations.mean()
median_idle = all_idle_durations.median()
max_idle = all_idle_durations.max()

# ------------------------- Dashboard -------------------------

# Section 1: Hourly Rentals (Bar Chart)
st.subheader("📊 Hourly Rentals: Weekday vs Weekend")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=hourly_rentals, x="hour", y="bike_id", hue="day_type", ax=ax1)
    ax1.set_title("Hourly Bike Rentals – Weekday vs Weekend")
    ax1.set_ylabel("Unique Rentals")
    ax1.set_xlabel("Hour of Day")
    st.pyplot(fig1)

# Section 2: Long Idle Stations
with col2:
    st.subheader("⏱️ Stations Where Bikes Sit Idle Too Long")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=idle_by_station.values, y=idle_by_station.index, palette="magma", ax=ax2)
    ax2.set_title("Top 10 Stations with Longest Average Idle Times")
    ax2.set_xlabel("Avg Idle Duration (hours)")
    ax2.set_ylabel("Station Name")
    st.pyplot(fig2)

# Section 3: Idle Duration Metrics
st.subheader("⏳ Typical Bike Parked Duration Before Reuse")
st.markdown("### 🧾 Summary Metrics")

col3, col4, col5 = st.columns(3)
col3.metric("⏱️ Average Idle Duration", f"{avg_idle:.2f} hrs")
col4.metric("📊 Median Idle Duration", f"{median_idle:.2f} hrs")
col5.metric("🚲 Maximum Idle Duration", f"{max_idle:.2f} hrs")

