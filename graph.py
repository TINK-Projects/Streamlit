import streamlit as st
import pandas as pd
import json
import pydeck as pdk
from datetime import datetime

# Set page config
st.set_page_config(layout="wide", page_title="Bike Activity Dashboard")

# Load JSONL data
with open("konstanz_march_2025.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort and calculate movement
df = df.sort_values(by=['bike_id', 'timestamp'])
df['prev_place'] = df.groupby('bike_id')['place_name'].shift()
df['movement'] = (df['place_name'] != df['prev_place']).fillna(False).astype(int)

# Total movements per bike
bike_movements = df.groupby('bike_id')['movement'].sum().reset_index(name='total_movements')

# Get latest location and timestamps
bike_info = df.groupby('bike_id').agg({
    'place_name': 'last',
    'lat': 'last',
    'lng': 'last',
    'timestamp': ['min', 'max']
}).reset_index()
bike_info.columns = ['bike_id', 'place_name', 'lat', 'lng', 'start_time', 'end_time']

# Merge and calculate idle span
bike_summary = pd.merge(bike_movements, bike_info, on='bike_id')
bike_summary['idle_span_hrs'] = (bike_summary['end_time'] - bike_summary['start_time']).dt.total_seconds() / 3600

# Filter unused bikes (0 or 1 movement)
unused_bikes = bike_summary[bike_summary['total_movements'] <= 1]

# Aggregate by station
unused_summary = unused_bikes.groupby(['place_name', 'lat', 'lng']).agg({
    'bike_id': 'count',
    'idle_span_hrs': 'mean'
}).reset_index().rename(columns={
    'bike_id': 'unused_bike_count',
    'idle_span_hrs': 'avg_idle_hours'
})

# Top 10
top10 = unused_summary.sort_values(by='unused_bike_count', ascending=False).head(10)

# Section Title
st.markdown("### 🗺️ Top 10 Stations with Most Unused Bikes")

# PyDeck map
layer = pdk.Layer(
    "ScatterplotLayer",
    data=top10,
    get_position='[lng, lat]',
    get_fill_color='[255, 0, 0, 140]',
    get_radius='unused_bike_count * 150',
    pickable=True,
    tooltip=True
)

view = pdk.ViewState(
    latitude=top10['lat'].mean(),
    longitude=top10['lng'].mean(),
    zoom=12,
    pitch=0
)

tooltip = {
    "html": "<b>{place_name}</b><br>🛑 Unused Bikes: {unused_bike_count}<br>⏱️ Avg Idle: {avg_idle_hours:.1f} hrs",
    "style": {"backgroundColor": "black", "color": "white"}
}

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=view,
    layers=[layer],
    tooltip=tooltip
))
