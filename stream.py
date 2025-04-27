import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import json

st.set_page_config(layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    with open("konstanz_march_2025.jsonl", "r") as file:
        records = [json.loads(line) for line in file]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(['bike_id', 'timestamp'], inplace=True)
    return df

df = load_data()

# ─── Prepare Movement Data ───────────────────────────────────────────────────────
df['prev_place'] = df.groupby('bike_id')['place_name'].shift(1)
movement_df = df[df['place_name'] != df['prev_place']].dropna(subset=['prev_place'])
movement_df['hour'] = movement_df['timestamp'].dt.hour
movement_df['day_type'] = movement_df['timestamp'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
rentals_per_hour = movement_df.groupby(['hour', 'day_type'])['bike_id'].nunique().reset_index()
rentals_pivot = rentals_per_hour.pivot(index='hour', columns='day_type', values='bike_id').fillna(0)

# ─── Idle Duration Metrics ──────────────────────────────────────────────────────
bike_idle_durations = []

for bike_id, group in df.groupby('bike_id'):
    group = group.sort_values('timestamp')
    group['prev_time'] = group['timestamp'].shift(1)
    group['prev_place'] = group['place_name'].shift(1)
    group = group[group['place_name'] == group['prev_place']]
    group['idle_duration'] = (group['timestamp'] - group['prev_time']).dt.total_seconds() / 3600
    bike_idle_durations.extend(group['idle_duration'].dropna().tolist())

avg_idle = round(pd.Series(bike_idle_durations).mean(), 2)
med_idle = round(pd.Series(bike_idle_durations).median(), 2)
max_idle = round(pd.Series(bike_idle_durations).max(), 2)

# ─── Idle Duration by Station ───────────────────────────────────────────────────
idle_station_data = []

for (bike_id, group) in df.groupby('bike_id'):
    group = group.sort_values('timestamp')
    group['prev_time'] = group['timestamp'].shift(1)
    group['prev_place'] = group['place_name'].shift(1)
    group = group[group['place_name'] == group['prev_place']]
    group['idle_duration'] = (group['timestamp'] - group['prev_time']).dt.total_seconds() / 3600
    idle_station_data.extend(group[['place_name', 'idle_duration']].dropna().values.tolist())

idle_df = pd.DataFrame(idle_station_data, columns=['place_name', 'idle_duration'])
station_avg_idle = idle_df.groupby('place_name')['idle_duration'].mean().sort_values(ascending=False).head(10)

# ─── Most Unused Bikes by Station ───────────────────────────────────────────────
latest_snapshot = df[df['timestamp'] == df['timestamp'].max()]
bike_counts = latest_snapshot['place_name'].value_counts().reset_index()
bike_counts.columns = ['place_name', 'idle_bike_count']
top_idle_stations = bike_counts.head(10)
station_coords = latest_snapshot.groupby('place_name')[['lat', 'lng']].mean().reset_index()
top_idle_stations = pd.merge(top_idle_stations, station_coords, on='place_name')

# ─── Station Activity ───────────────────────────────────────────────────────────
top_departure_stations = movement_df['prev_place'].value_counts().head(5)
top_arrival_stations = movement_df['place_name'].value_counts().head(5)

# ─── Dashboard Layout ───────────────────────────────────────────────────────────

st.title("🚴‍♂️ Bike Activity Dashboard – Konstanz (March 2025)")

# ─── Key Insights ───────────────────────────────────────────────────────────────
st.info(f"""
**Key Insights:**
- Average idle time before reuse is **{avg_idle} hours**.
- Peak rental hour is **{rentals_pivot.sum(axis=1).idxmax()}00h**.
- Top station with most idle bikes: **{top_idle_stations.iloc[0]['place_name']}**.
""")

# ─── Idle Summary Metrics ───────────────────────────────────────────────────────
with st.container():
    st.subheader("⏳ Typical Bike Parked Duration Before Reuse")
    col1, col2, col3 = st.columns(3)
    col1.metric("🕒 Avg Idle Duration", f"{avg_idle} hrs")
    col2.metric("📊 Median Idle Duration", f"{med_idle} hrs")
    col3.metric("🚲 Max Idle Duration", f"{max_idle} hrs")

# ─── Rentals Chart and Most Rented Bike Type Chart ─────────────────────────────
with st.container():
    st.subheader("📊 Bike Usage Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hourly Rentals: Weekday vs Weekend")
        fig, ax = plt.subplots(figsize=(6, 4))
        rentals_pivot.plot(kind='bar', ax=ax)
        ax.set_ylabel("Unique Rentals")
        ax.set_xlabel("Hour of Day")
        ax.set_title("Hourly Bike Rentals")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Most Rented Bikes by Type")
        bike_type_mapping = {
            299: "Standard bike",
            300: "E-bike",
            301: "Cargo bike",
            305: "Special bike"
        }
        rented_bikes = movement_df[['bike_id']].drop_duplicates().merge(df[['bike_id', 'bike_type']].drop_duplicates(), on='bike_id')
        rented_bike_type_counts = rented_bikes['bike_type'].map(bike_type_mapping).value_counts()

        fig3, ax3 = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax3.pie(
            rented_bike_type_counts,
            labels=None,
            autopct=lambda pct: ('%.1f%%' % pct) if pct > 3 else '',
            startangle=90,
            pctdistance=0.7,
            textprops={'fontsize': 8}
        )
        centre_circle = plt.Circle((0, 0), 0.60, fc='white')
        fig3.gca().add_artist(centre_circle)
        ax3.set_title("Most Rented Bike Types", fontsize=12)
        ax3.legend(rented_bike_type_counts.index, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3)

# ─── Top Pickup and Dropoff Stations ───────────────────────────────────────────
with st.container():
    st.subheader("🚚 Top 5 Pickup and Drop-off Stations")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Pickups (Departures)")
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        top_departure_stations = top_departure_stations.reset_index()
        top_departure_stations.columns = ['Station', 'Pickups']
        sns.barplot(
            y='Station',
            x='Pickups',
            data=top_departure_stations,
            ax=ax4,
            palette='Blues_d',
            saturation=1,
            linewidth=0,
            width=0.5
        )
        for patch in ax4.patches:
            patch.set_edgecolor('none')
        for i in range(len(top_departure_stations)):
            ax4.text(top_departure_stations['Pickups'][i] + 1, i, top_departure_stations['Pickups'][i], va='center')
        ax4.set_xlabel("")
        ax4.set_ylabel("")
        ax4.set_xticks([])
        ax4.set_title("Top Pickup Stations")
        st.pyplot(fig4)

    with col2:
        st.subheader("Top Drop-offs (Arrivals)")
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        top_arrival_stations = top_arrival_stations.reset_index()
        top_arrival_stations.columns = ['Station', 'Drop-offs']
        sns.barplot(
            y='Station',
            x='Drop-offs',
            data=top_arrival_stations,
            ax=ax5,
            palette='Greens_d',
            saturation=1,
            linewidth=0,
            width=0.5
        )
        for patch in ax5.patches:
            patch.set_edgecolor('none')
        for i in range(len(top_arrival_stations)):
            ax5.text(top_arrival_stations['Drop-offs'][i] + 1, i, top_arrival_stations['Drop-offs'][i], va='center')
        ax5.set_xlabel("")
        ax5.set_ylabel("")
        ax5.set_xticks([])
        ax5.set_title("Top Drop-off Stations")
        st.pyplot(fig5)


# ─── Map of Idle Bikes ─────────────────────────────────────────────────────────
with st.container():
    st.subheader("🗺️ Top 10 Stations with Most Unused Bikes")
    st.dataframe(top_idle_stations[['place_name', 'idle_bike_count']])
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        initial_view_state=pdk.ViewState(
            latitude=47.66,
            longitude=9.17,
            zoom=13,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=top_idle_stations,
                get_position='[lng, lat]',
                get_color='[255, 0, 0, 160]',
                get_radius='idle_bike_count * 20',
                pickable=True
            )
        ],
        tooltip={"text": "{place_name}\nIdle Bikes: {idle_bike_count}"}
    ))
