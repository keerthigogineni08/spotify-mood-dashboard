import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

# Load Data
data = pd.read_csv("data/data_w_genres.csv")
data_1m = pd.read_csv("data/spotify_data.csv")
data_2024 = pd.read_csv("data/Most Streamed Spotify Songs 2024.csv", encoding="ISO-8859-1")

st.title("🎵 Spotify Mood Dashboard")

# Sidebar track selection
st.sidebar.header("Track Explorer")
all_tracks = data_1m['track_name'].dropna().unique()
selected_track = st.sidebar.selectbox("Choose a track to explore: ", sorted(all_tracks))

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
ax.set_title("🎯 Mood Clusters (PCA + KMeans)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)

# ================== Mood Map (Valence vs Energy) ==================
st.subheader("🎨 Mood Map: Valence vs Energy by Genre")
try:
    import plotly.express as px
    fig = px.scatter(
        data_frame=data,
        x="valence",
        y="energy",
        color="genres",
        hover_data=["artists"],
        title="🎨 Mood Map: Valence vs Energy by Genre"
    )
    st.plotly_chart(fig)
except ImportError:
    st.warning("Plotly not installed. Falling back to matplotlib.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x="valence", y="energy", hue="genres", alpha=0.6, legend=False, ax=ax)
    ax.set_title("Mood Map: Valence vs Energy")
    st.pyplot(fig)

# ================== Popularity Prediction ==================
st.subheader("🔮 Predicting Popularity (ML Model)")
X_pop = data_1m[features].dropna()
y_pop = data_1m.loc[X_pop.index, 'popularity']

X_train, X_test, y_train, y_test = train_test_split(X_pop, y_pop, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
st.write(f"RMSE of the model: {rmse:.2f}")

# ================== 2024 Highlights ==================
st.subheader("🔥 Most Streamed Songs of 2024")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=data_2024,
    x="Track Score",
    y="Spotify Streams",
    hue="Artist",
    alpha=0.7,
    legend=False,
    ax=ax
)
ax.set_title("🔥 Most Streamed Songs of 2024")
ax.set_xlabel("Track Score")
ax.set_ylabel("Spotify Streams")
st.pyplot(fig)
