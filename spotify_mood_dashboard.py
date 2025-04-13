import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.express as px
import streamlit as st
import requests
from urllib.parse import quote
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from urllib.parse import quote
from math import pi
import random

# Get credentials from Streamlit secrets
client_id = st.secrets["spotify_client_id"]
client_secret = st.secrets["spotify_client_secret"]

# Step 1: Get an access token from Spotify API
def get_spotify_token():
    auth_url = "https://accounts.spotify.com/api/token"
    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    })
    if auth_response.status_code == 200:
        return auth_response.json().get("access_token")
    return None

# Step 2: Search for a track by name (and optionally artist)
def search_spotify_track(track_name, artist_name=None):
    token = get_spotify_token()
    if not token:
        return None

    headers = {
        "Authorization": f"Bearer {token}"
    }

    query = f"track:{track_name}"
    if artist_name:
        query += f" artist:{artist_name}"
    query = quote(query)

    search_url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1"
    response = requests.get(search_url, headers=headers)

    if response.status_code == 200:
        items = response.json().get("tracks", {}).get("items", [])
        if items:
            return items[0]["id"]  # Return Spotify track ID
    return None


# Set page layout
st.set_page_config(layout="wide", page_title="Spotify Mood Dashboard", page_icon="🎵")
sns.set(style="whitegrid")

def deduplicate_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    return new_cols

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

    # Load and preprocess regional CSVs
    language_dfs = []
    language_sources = {
        "Assamese": "data/archive/Assamese_songs.csv",
        "Bengali": "data/archive/Bengali_songs.csv",
        "Bhojpuri": "data/archive/Bhojpuri_songs.csv",
        "Gujarati": "data/archive/Gujarati_songs.csv",
        "Haryanvi": "data/archive/Haryanvi_songs.csv",
        "Hindi": "data/archive/Hindi_songs.csv",
        "Kannada": "data/archive/Kannada_songs.csv",
        "Malayalam": "data/archive/Malayalam_songs.csv",
        "Marathi": "data/archive/Marathi_songs.csv",
        "Odia": "data/archive/Odia_songs.csv",
        "Old": "data/archive/Old_songs.csv",
        "Punjabi": "data/archive/Punjabi_songs.csv",
        "Rajasthani": "data/archive/Rajasthani_songs.csv",
        "Tamil": "data/archive/Tamil_songs.csv",
        "Telugu": "data/archive/Telugu_songs.csv",
        "Urdu": "data/archive/Urdu_songs.csv"
    }
    for lang, path in language_sources.items():
        if os.path.exists(path):
            df = pd.read_csv(path)

            # Rename common columns to match rest of dashboard
            df = df.rename(columns={
                'song_name': 'track_name',
                'singer': 'artist_name',
                'Valence': 'valence',
                'duration': 'duration_str'
            })

            # Convert mm:ss to milliseconds
            df['duration_ms'] = df['duration_str'].apply(
                lambda x: int(x.split(':')[0]) * 60000 + int(x.split(':')[1]) * 1000
                if isinstance(x, str) and ':' in x else None
            )

            df['language'] = lang

            # Drop unnecessary or renamed columns
            df = df.drop(columns=['duration_str'], errors='ignore')

            # Make column names lowercase and deduplicate if needed
            df.columns = [col.lower() for col in df.columns]
            df.columns = deduplicate_columns(df.columns)


            # Append clean df to list
            language_dfs.append(df)

    # Load and clean the Telugu XLSX file
    telugu_xlsx_path = "data/Spotify Telugu.xlsx"
    if os.path.exists(telugu_xlsx_path):
        telugu_xlsx = pd.read_excel(telugu_xlsx_path)
        telugu_xlsx = telugu_xlsx.rename(columns={
            'Name': 'track_name',
            'Artists': 'artist_name',
            'Valence': 'valence',
            'Intrumentalness': 'instrumentalness',
            'Duration': 'duration_ms'
        })
        telugu_xlsx.columns = [col.lower() for col in telugu_xlsx.columns]
        telugu_xlsx['language'] = 'Telugu'
        language_dfs.append(telugu_xlsx)

    # Combine all regional data
    if language_dfs:
        regional_data = pd.concat(language_dfs, ignore_index=True)

        # ✅ Tag all rows without language as 'English'
        if 'language' not in data_1m.columns:
            data_1m['language'] = 'English'
        else:
            data_1m['language'] = data_1m['language'].fillna('English')

        data_1m = pd.concat([data_1m, regional_data], ignore_index=True, sort=False)
        st.sidebar.caption(f"🌍 Loaded {regional_data.shape[0]} regional songs across all languages.")


# ===================== 🔧 Preprocessing =====================
# Clean 'data' (the file with genres)
data['genres'] = data['genres'].apply(lambda x: eval(x) if isinstance(x, str) else [])
data = data.explode('genres')

# Standardize artist column name for merging
data.rename(columns={'artists': 'artist_name'}, inplace=True)

# Drop old genre column from data_1m if present
if 'genre' in data_1m.columns:
    data_1m.drop(columns=['genre'], inplace=True)

# Merge genres from data into data_1m using artist_name
if 'genres' not in data_1m.columns:
    data_1m = data_1m.merge(
        data[['artist_name', 'genres']],
        on='artist_name',
        how='left'
    )
    data_1m['genres'] = data_1m['genres'].fillna("Unknown")
    data_1m = data_1m.explode('genres')

st.title("🎵 Spotify Mood Dashboard")

# ===================== Language Filter =====================

available_languages = data_1m['language'].dropna().unique() if 'language' in data_1m.columns else []

#st.sidebar.write("Languages found in data_1m:", data_1m['language'].unique() if 'language' in data_1m.columns else "No language column")

if len(available_languages) > 0:
    selected_language = st.sidebar.selectbox("🌍 Filter by Language", ["All"] + sorted(available_languages))
    if selected_language != "All":
        data_1m = data_1m[data_1m['language'] == selected_language]

# ===================== 🔍 Sidebar Filters =====================
st.sidebar.header("Track Explorer")
all_tracks = data_1m['track_name'].dropna().unique()
search_query = st.sidebar.text_input("🔎 Search for a track", "")
filtered_tracks = [track for track in all_tracks if search_query.lower() in track.lower()]
selected_track = st.sidebar.selectbox("Choose a track to explore: ", filtered_tracks if filtered_tracks else all_tracks)

st.sidebar.markdown("---")
genres = sorted(data_1m['genres'].dropna().unique())
selected_genres = st.sidebar.multiselect("🎵 Filter by Genre", genres)

if st.sidebar.button("🔁 Reset Genre Filters"):
    selected_genres = []

if selected_genres:
    data = data[data['genres'].isin(selected_genres)]
    data_1m = data_1m[data_1m['genres'].isin(selected_genres)]

# ===================== 🌈 Radar Chart =====================
st.subheader("🌟 Track Mood Breakdown (Radar Chart)")
st.caption("Understand how a selected song scores across musical moods like danceability, valence, and energy.")

def plot_radar_interactive(track_name):
    track = data_1m[data_1m['track_name'] == track_name]
    if track.empty:
        st.warning(f"Track '{track_name}' not found.")
        return

    row = track.iloc[0]
    labels = ['danceability', 'energy', 'acousticness', 'valence', 'liveness', 'instrumentalness']
    values = [row[label] for label in labels]

    mean_vals = data_1m[labels].mean()
    std_vals = data_1m[labels].std()
    z_scores = [(row[label] - mean_vals[label]) / std_vals[label] for label in labels]

    labels += [labels[0]]
    z_scores += [z_scores[0]]

    fig = px.line_polar(
        r=z_scores,
        theta=labels,
        line_close=True,
        title=f"🎯 Mood Breakdown: {track_name} 💫",
    )
    fig.update_traces(fill='toself', line_color="#FF4B4B", opacity=0.8)
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-2, 2])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

plot_radar_interactive(selected_track)

# ===================== 🔍 Mood Clusters =====================
st.subheader("🧠 Mood Clusters Explained (PCA + KMeans)")
st.markdown("Each dot is a song. We've grouped similar moods using AI. Colors = Vibes! 🎨")

features = ['valence', 'energy', 'danceability', 'acousticness']
X = data_1m[features].dropna()

if X.empty:
    st.warning("Not enough data to cluster moods. Try selecting more genres or resetting filters.")
else:
    # Filter matching rows
    filtered_data = data_1m.loc[X.index].copy()

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

    filtered_data['Cluster'] = clusters
    filtered_data['Mood'] = filtered_data['Cluster'].map(mood_labels)

    fig_clusters = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=filtered_data['Mood'],
        labels={'x': 'PCA 1', 'y': 'PCA 2'},
        title="🧠 Mood Clusters (AI-generated) 🎨",
        opacity=0.7,
        width=1000
    )
    fig_clusters.update_traces(marker=dict(size=5))
    st.plotly_chart(fig_clusters, use_container_width=True)

    with st.expander("ℹ️ What are mood clusters?"):
        st.markdown("We used PCA + KMeans to group songs with similar moods. This helps us identify types of songs based on feel, not just genre.")

    st.markdown("## 🎨 AI Mood Clusters")
    st.caption("Songs grouped using AI based on sound similarity. Each color is a mood category like Chill, Sad Bops, or Hype.")

    # Optional: merge moods back into full data_1m (so other features like "Surprise Me" and recommender work)
    data_1m = data_1m.merge(
        filtered_data[['track_name', 'artist_name', 'Mood', 'Cluster']],
        on=['track_name', 'artist_name'],
        how='left'
    )

# ===================== 🎨 Mood Map =====================
st.subheader("🎨 Mood Map: Valence vs Energy by Mood (Interactive)")
st.markdown("This chart maps songs by **happiness (valence)** vs **intensity (energy)**. Each dot is a song, color = mood 🎨")

try:
    plot_data = data_1m.sample(n=1000, random_state=42) if len(data_1m) > 1000 else data_1m
    fig_mood_map = px.scatter(
        data_frame=plot_data,
        x="valence",
        y="energy",
        color="Mood",
        hover_data=["artist_name", "track_name"],
        title="🎨 Mood Map: Valence vs Energy by Mood",
    )
    #st.image("https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif", width=150)
    st.plotly_chart(fig_mood_map, use_container_width=True)
except Exception as e:
    st.warning("Could not load Plotly chart. Showing fallback.")
    try:
        fig_fallback, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data_1m, x="valence", y="energy", hue="Mood", alpha=0.6, ax=ax, legend=False)
        ax.set_title("Mood Map")
        ax.set_xlabel("Valence")
        ax.set_ylabel("Energy")
        st.pyplot(fig_fallback)
    except Exception as fallback_error:
        st.error(f"Both Plotly and Matplotlib failed. Error: {fallback_error}")


# ===================== 🎉 Fun Visual =====================
st.subheader("🎉 Enjoy the Vibes!")

gif_map = {
    "Chill & Mellow": "https://media.giphy.com/media/3o7aCTfyhYawdOXcFW/giphy.gif",
    "Party Hype": "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif",
    "Sad Bops": "https://media.giphy.com/media/l0HlHFRbmaZtBRhXG/giphy.gif",
    "Confident Bangers": "https://media.giphy.com/media/3o6ZsX2PUNLpmj7pQI/giphy.gif",
    "Acoustic Vibes": "https://media.giphy.com/media/d2lcHJTG5Tscg/giphy.gif"
}

if selected_track and 'Mood' in data_1m.columns:
    mood_row = data_1m[data_1m['track_name'] == selected_track]
    if not mood_row.empty:
        mood = mood_row['Mood'].values[0]
    else:
        mood = "Chill & Mellow"
else:
    mood = "Chill & Mellow"

st.markdown(f"🎶 You're vibing with **{mood}** mood today!")
#st.image(gif_map.get(mood, gif_map["Chill & Mellow"]), caption=f"You're vibing with {mood} 🎶", use_container_width=True)

# ===================== 🔮 Popularity Prediction =====================
st.subheader("🔮 Predicting Popularity (ML Model)")

def search_spotify_track(track_name, artist_name=None):
    token = get_spotify_token()
    if not token:
        return None, None

    headers = {"Authorization": f"Bearer {token}"}
    query = f"track:{track_name}"
    if artist_name:
        query += f" artist:{artist_name}"
    query = quote(query)

    url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        items = response.json().get("tracks", {}).get("items", [])
        if items:
            return items[0]["id"], items[0]["popularity"]
    return None, None

try:
    X_pop = data_1m[features].dropna()
    y_pop = data_1m.loc[X_pop.index, 'popularity']

    if y_pop.isnull().any():
        st.warning("⚠️ Some songs are missing popularity scores. Popularity prediction skipped.")
        model = None
    else:
        with st.spinner("Training model..."):
            X_train, X_test, y_train, y_test = train_test_split(X_pop, y_pop, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds) ** 0.5

        st.success(f"🎯 Popularity Prediction RMSE: {rmse:.2f}")


    if model is not None:
        st.subheader("🎯 Popularity Prediction Demo")
        valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
        sample_input = pd.DataFrame([[valence, energy, danceability, acousticness]], columns=features)
        prediction = model.predict(sample_input)[0]
        st.success(f"🎷 Predicted Popularity: **{prediction:.2f}**")

        if prediction > 80:
            st.balloons()
        elif prediction < 30:
            st.snow()

except Exception as e:
    st.error(f"Error in popularity prediction: {e}")

# ===================== 🔥 Top Songs of 2024 =====================
st.subheader("🔥 Top 20 Streamed Songs of 2024")
try:
    top_20 = data_2024.sort_values("Spotify Streams", ascending=False).head(20)
    fig = px.bar(
        top_20,
        x="Spotify Streams",
        y="Track",
        color="Artist",
        orientation="h",
        title="🔥 Top 20 Most Streamed Songs of 2024",
        text="Spotify Streams",
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Couldn't load top 20 chart: {e}")

# ===================== 🧠 Smart Mood Recommender =====================
st.subheader("🧠 Smart Mood Recommender")

col1, col2 = st.columns(2)
with col1:
    fav_genre = st.selectbox("🎧 Select your favorite genre:", genres)
with col2:
    mood_pick = st.select_slider("🎭 Pick your mood preference:", options=["Chill", "Sad", "Energetic", "Happy", "Mellow"])

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

if recommendations.empty:
    st.info("🙁 No matching songs found. Try adjusting your mood or genre.")

spotify_base = "https://open.spotify.com/track/"

for i, row in recommendations.iterrows():
    track_name = row["track_name"]
    artist_name = row.get("artist_name")
    
    # Fetch from API if track_id is missing
    track_id = row.get("track_id") if "track_id" in row and pd.notnull(row["track_id"]) else None
    spotify_pop = row.get("popularity") if pd.notnull(row.get("popularity")) else None

    if not track_id:
        track_id, spotify_pop = search_spotify_track(track_name, artist_name)

    pop_display = spotify_pop if spotify_pop is not None else "Unknown"

    if track_id:
        spotify_link = f"https://open.spotify.com/track/{track_id}"
        st.markdown(f"**{track_name}** by *{artist_name}* — Popularity: {pop_display} [🎧 Open on Spotify]({spotify_link})")
    else:
        st.markdown(f"**{track_name}** by *{artist_name}* — Popularity: {pop_display}")

# ===================== 🎲 Surprise Me =====================
if st.button("🎲 Surprise Me with a Track!"):
    surprise = data_1m.sample(1).iloc[0]
    st.info(f"🎵 Track: **{surprise['track_name']}** by *{surprise['artist_name']}* | Genre: {surprise['genres']} | Popularity: {surprise['popularity']}")
    if surprise['popularity'] > 80:
        st.balloons()
    elif surprise['popularity'] < 30:
        st.snow()

# ===================== 🎨 Theme Instructions =====================
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
