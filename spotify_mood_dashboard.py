import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import pi

# Set page layout
st.set_page_config(layout="wide", page_title="Spotify Mood Dashboard", page_icon="🎵")
sns.set(style="whitegrid")

# Load Data with spinner
with st.spinner("Loading data..."):
    data = pd.read_csv("data/data_w_genres.csv")
    data_2024 = pd.read_csv("data/Most Streamed Spotify Songs 2024.csv", encoding="ISO-8859-1")

    if os.path.exists("data/spotify_data.csv"):
        data_1m = pd.read_csv("data/spotify_data.csv")
        st.sidebar.success("✅ Loaded full dataset (1M tracks)")
    else:
        data_1m = pd.read_csv("data/spotify_data_sample.csv")
        st.sidebar.warning("⚠️ Using sample dataset (10k tracks) for demo")

st.title("🎵 Spotify Mood Dashboard")

# Sidebar: Search and Filters
st.sidebar.header("Track Explorer")
all_tracks = data_1m['track_name'].dropna().unique()
search_query = st.sidebar.text_input("🔎 Search for a track", "")
filtered_tracks = [track for track in all_tracks if search_query.lower() in track.lower()]
selected_track = st.sidebar.selectbox("Choose a track to explore: ", filtered_tracks if filtered_tracks else all_tracks)

# Genre filter
st.sidebar.markdown("---")
genres = data['genres'].dropna().unique()
selected_genres = st.sidebar.multiselect("🎵 Filter by Genre", genres)

if selected_genres:
    data = data[data['genres'].isin(selected_genres)]
    data_1m = data_1m[data_1m['genres'].isin(selected_genres)]

# ================== Radar Chart ==================
st.subheader("🌟 Track Mood Breakdown (Radar Chart)")
def plot_radar(track_name):
    track = data_1m[data_1m['track_name'] == track_name]
    if track.empty:
        st.warning(f"Track '{track_name}' not found.")
        return

    row = track.iloc[0]
    labels = ['danceability', 'energy', 'acousticness', 'valence', 'liveness', 'instrumentalness']
    values = [row[label] for label in labels]

    mean_vals = data_1m[labels].mean()
    std_vals = data_1m[labels].std()
    values = [(row[label] - mean_vals[label]) / std_vals[label] for label in labels]

    values += values[:1]  # loop
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(f"Radar Chart: {track_name}")
    st.pyplot(fig)

plot_radar(selected_track)

# ================== Fun Visual ==================
st.subheader("🎉 Enjoy the Vibes!")
st.image("https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy.gif", caption="You're vibing with Spotify Moods 🎶", use_column_width=True)

# ================== Mood Clustering ==================
st.subheader("🔍 Mood Clusters (PCA + KMeans)")
features = ['valence', 'energy', 'danceability', 'acousticness']
X = data_1m[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_pca)

mood_labels = {
    0: "Chill & Mellow",
    1: "Party Hype",
    2: "Sad Bops",
    3: "Confident Bangers",
    4: "Acoustic Vibes"
}
data_1m['Cluster'] = clusters
data_1m['Mood'] = data_1m['Cluster'].map(mood_labels)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data_1m['Mood'], palette='Set2', alpha=0.6, ax=ax)
ax.set_title("🌟 Mood Clusters (PCA + KMeans)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)

# ================== Mood Map (Valence vs Energy) ==================
st.subheader("🎨 Mood Map: Valence vs Energy by Genre (Interactive)")
try:
    fig = px.scatter(
        data_frame=data,
        x="valence",
        y="energy",
        color="genres",
        hover_data=["artists"],
        title="🎨 Mood Map: Valence vs Energy by Genre",
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning("Could not load Plotly chart. Showing fallback.")
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x="valence", y="energy", hue="genres", alpha=0.6, legend=False)
        plt.title("Mood Map")
        plt.xlabel("Valence")
        plt.ylabel("Energy")
        st.pyplot()
    except Exception as fallback_error:
        st.error(f"Both Plotly and Matplotlib failed. Error: {fallback_error}")

# =================== Predicting Popularity ===================
st.subheader("🔮 Predicting Popularity (ML Model)")
try:
    X_pop = data_1m[features].dropna()
    y_pop = data_1m.loc[X_pop.index, 'popularity']

    with st.spinner("Training model..."):
        X_train, X_test, y_train, y_test = train_test_split(X_pop, y_pop, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5

    st.success(f"🎯 Popularity Prediction RMSE: {rmse:.2f}")

    # Live prediction demo
    st.subheader("🎯 Popularity Prediction Demo")
    st.markdown("Adjust sliders below to simulate a track's mood:")
    valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    sample_input = pd.DataFrame([[valence, energy, danceability, acousticness]], columns=features)
    prediction = model.predict(sample_input)[0]
    st.success(f"🎷 Predicted Popularity: **{prediction:.2f}**")

except Exception as e:
    st.error(f"Error in popularity prediction: {e}")


# ================== Top 20 Streamed Songs of 2024 ==================
st.subheader("🔥 Top 20 Streamed Songs of 2024")

top_20 = data_2024.sort_values("Spotify Streams", ascending=False).head(20)

fig = px.bar(
    top_20,
    x="Spotify Streams",
    y="Track Name",
    color="Artist",
    orientation="h",
    title="🔥 Top 20 Most Streamed Songs of 2024",
    text="Spotify Streams",
)

fig.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig, use_container_width=True)

# ================== Smart Mood Recommender ==================
st.subheader("🧠 Smart Mood Recommender")

col1, col2 = st.columns(2)
with col1:
    fav_genre = st.selectbox("🎧 Select your favorite genre:", sorted(data_1m['genres'].dropna().unique()))
with col2:
    mood_pick = st.select_slider("🎭 Pick your mood preference:", options=["Chill", "Sad", "Energetic", "Happy", "Mellow"])

# Map mood pick to valence/energy range
mood_filters = {
    "Chill": (0.2, 0.5),
    "Sad": (0.0, 0.4),
    "Energetic": (0.7, 1.0),
    "Happy": (0.6, 1.0),
    "Mellow": (0.3, 0.6)
}

val_min, val_max = mood_filters[mood_pick]

recommendations = data_1m[
    (data_1m['genres'] == fav_genre) &
    (data_1m['valence'].between(val_min, val_max))
].sort_values("popularity", ascending=False).head(5)

if not recommendations.empty:
    st.success("🎯 Based on your mood and genre, here are some recommendations:")
    for i, row in recommendations.iterrows():
        st.markdown(f"**{row['track_name']}** by *{row['artists']}* — Popularity: {row['popularity']}")
else:
    st.info("🙁 No matching songs found. Try adjusting your mood or genre.")


# ================== Theme Setup Instructions ==================
with st.expander("🎨 How to Apply Streamlit Theme"):
    st.code("""
# Inside .streamlit/config.toml
[theme]
base="dark"
primaryColor="#1DB954"
backgroundColor="#121212"
secondaryBackgroundColor="#191414"
textColor="#FFFFFF"
    """)
