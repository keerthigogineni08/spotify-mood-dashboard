# ===================== 1. Imports & Setup =====================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import requests
import random
from urllib.parse import quote
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from urllib.parse import quote

# Set up Streamlit page config
st.set_page_config(layout="wide", page_title="Spotify Mood Dashboard", page_icon="🎵")
sns.set(style="whitegrid")

# ===================== 2. Spotify Auth & Utility Functions  =====================
@st.cache_data(ttl=3600)
def get_spotify_token():
    auth_url = "https://accounts.spotify.com/api/token"
    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': st.secrets["spotify_client_id"],
        'client_secret': st.secrets["spotify_client_secret"],
    })
    if auth_response.status_code == 200:
        return auth_response.json().get("access_token")
    return None

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
            return items[0].get("id"), items[0].get("popularity")
    return None, None

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

# ===================== 3. Load and Merge All Data =====================
st.sidebar.header("📂 Data Loading")
with st.spinner("Loading and preprocessing datasets..."):
    data_genre = pd.read_csv("data/data_w_genres.csv")
    data_2024 = pd.read_csv("data/Most Streamed Spotify Songs 2024.csv", encoding="ISO-8859-1")

    if os.path.exists("data/spotify_data.csv"):
        data_main = pd.read_csv("data/spotify_data.csv")
        st.sidebar.success("✅ Loaded full dataset (1M tracks)")
    else:
        data_main = pd.read_csv("data/spotify_data_sample.csv")
        st.sidebar.warning("⚠️ Using sample dataset (10k tracks)")

    # Load regional CSVs
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
            df = df.rename(columns={
                'song_name': 'track_name',
                'singer': 'artist_name',
                'Valence': 'valence',
                'duration': 'duration_str'
            })
            df['duration_ms'] = df['duration_str'].apply(
                lambda x: int(x.split(':')[0]) * 60000 + int(x.split(':')[1]) * 1000
                if isinstance(x, str) and ':' in x else None
            )
            df['language'] = lang
            df = df.drop(columns=['duration_str'], errors='ignore')
            df.columns = [col.lower() for col in df.columns]
            df.columns = deduplicate_columns(df.columns)
            language_dfs.append(df)

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

    if language_dfs:
        regional_data = pd.concat(language_dfs, ignore_index=True)
        if 'language' not in data_main.columns:
            data_main['language'] = 'English'
        else:
            data_main['language'] = data_main['language'].fillna('English')

        data_main = pd.concat([data_main, regional_data], ignore_index=True, sort=False)
        st.sidebar.caption(f"🌍 Loaded {regional_data.shape[0]} regional songs across all languages.")

    # Temporary empty fallback for cleaned_data to avoid crash
    cleaned_data = pd.DataFrame(columns=['valence', 'energy', 'danceability', 'acousticness'])

# ===================== 4. Clean & Merge Genre Info =====================
data_genre['genres'] = data_genre['genres'].apply(lambda x: eval(x) if isinstance(x, str) else [])
data_genre = data_genre.explode('genres')
data_genre.rename(columns={'artists': 'artist_name'}, inplace=True)

if 'genre' in data_main.columns:
    data_main.drop(columns=['genre'], inplace=True)

if 'genres' not in data_main.columns:
    data_main = data_main.merge(
        data_genre[['artist_name', 'genres']],
        on='artist_name',
        how='left'
    )
    data_main['genres'] = data_main['genres'].fillna("Unknown")
    data_main = data_main.explode('genres')

# ===================== 5. Create cleaned_data for clustering, mood map, etc. =====================
cleaned_data = data_main[
    (data_main['valence'] > 0) &
    (data_main['energy'] > 0) &
    (data_main['danceability'] > 0) &
    (data_main['acousticness'] > 0)
].dropna(subset=['valence', 'energy', 'danceability', 'acousticness'])


# ===================== MAIN DASHBOARD WITH TABS =====================
main_tab, analysis_tab = st.tabs(["🎧 Mood Dashboard", "📊 Data Insights"])

# ===================== 🎧 Mood Dashboard =====================
with main_tab:
    st.title("🎧 Spotify Mood Dashboard")
    st.markdown("Welcome to the main experience! Explore moods, get song recommendations, and visualize music with AI.")

    # ===================== 1. Sidebar Filters & Track Selection =====================
    st.sidebar.header("🎛️ Track Explorer")

    available_languages = data_main['language'].dropna().unique() if 'language' in data_main.columns else []
    selected_language = st.sidebar.selectbox("🌍 Filter by Language", ["All"] + sorted(available_languages))
    if selected_language != "All":
        data_main = data_main[data_main['language'] == selected_language]

    all_tracks = data_main['track_name'].dropna().unique()
    search_query = st.sidebar.text_input("🔎 Search for a track", "")
    filtered_tracks = [track for track in all_tracks if search_query.lower() in track.lower()]
    selected_track = st.sidebar.selectbox("🎶 Choose a track to explore:", filtered_tracks if filtered_tracks else all_tracks)

    genres = sorted(data_main['genres'].dropna().unique())
    selected_genres = st.sidebar.multiselect("🎵 Filter by Genre", genres)
    if selected_genres:
        data_main = data_main[data_main['genres'].isin(selected_genres)]

    # ===================== 2. Radar Chart Visualization =====================
    st.title("🎵 Spotify Mood Dashboard")
    st.subheader("🌟 Track Mood Breakdown (Radar Chart)")
    st.caption("Understand how a selected song scores across musical moods like danceability, valence, and energy.")

    def plot_radar_interactive(track_name):
        track = data_main[data_main['track_name'] == track_name]
        if track.empty:
            st.warning(f"Track '{track_name}' not found.")
            return

        row = track.iloc[0]
        labels = ['danceability', 'energy', 'acousticness', 'valence', 'liveness', 'instrumentalness']
        values = [row.get(label, 0) for label in labels]

        mean_vals = data_main[labels].mean()
        std_vals = data_main[labels].std()
        z_scores = [(row[label] - mean_vals[label]) / std_vals[label] if std_vals[label] != 0 else 0 for label in labels]

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

    # ===================== 3. Popularity Model (Train Once) =====================
    st.subheader("🔮 Popularity Prediction (Model)")
    st.caption("This ML model is trained on danceability, valence, energy, acousticness to predict how popular a track might be.")

    features = ['valence', 'energy', 'danceability', 'acousticness']
    model = None

    try:
        # Use only rows where all features AND popularity are present
        valid_rows = data_main[features + ['popularity']].dropna()
        X_pop = valid_rows[features]
        y_pop = valid_rows['popularity']

        X_train, X_test, y_train, y_test = train_test_split(X_pop, y_pop, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        st.success(f"🎯 Model trained. RMSE: {rmse:.2f}")
    except Exception as e:
        st.error(f"Model training failed: {e}")


    # ===================== 4. Smart Mood Recommender =====================
    st.subheader("🧠 Smart Mood Recommender")

    col1, col2 = st.columns(2)
    with col1:
        fav_genre = st.selectbox("🎧 Select your favorite genre:", genres)
    with col2:
        mood_pick = st.select_slider("🎭 Pick your mood:", options=["Chill", "Sad", "Energetic", "Happy", "Mellow"])

    mood_filters = {
        "Chill": (0.2, 0.5),
        "Sad": (0.0, 0.4),
        "Energetic": (0.7, 1.0),
        "Happy": (0.6, 1.0),
        "Mellow": (0.3, 0.6)
    }
    val_min, val_max = mood_filters[mood_pick]

    recommendations = data_main[
        (data_main['genres'] == fav_genre) &
        (data_main['valence'].between(val_min, val_max))
    ].sort_values("popularity", ascending=False).head(5)

    if recommendations.empty:
        st.info("🙁 No matching songs found. Try adjusting your mood or genre.")

    for _, row in recommendations.iterrows():
        track_name = row["track_name"]
        artist_name = row.get("artist_name")
        track_id = row.get("track_id")
        spotify_pop = row.get("popularity")

        # Fallback: if track_id or popularity is missing, search or predict
        if not track_id:
            track_id, spotify_pop = search_spotify_track(track_name, artist_name)

        if not spotify_pop and model is not None:
            try:
                features_row = row[features]
                if features_row.notnull().all():
                    spotify_pop = model.predict(pd.DataFrame([features_row]))[0]
            except:
                spotify_pop = "Unknown"

        if track_id:
            st.markdown(f"**{track_name}** by *{artist_name}* — Popularity: {round(spotify_pop,2) if spotify_pop else 'Unknown'} [🎧 Open on Spotify](https://open.spotify.com/track/{track_id})")
        else:
            search_url = f"https://open.spotify.com/search/{quote(track_name)}"
            st.markdown(f"**{track_name}** by *{artist_name}* — Popularity: {spotify_pop or 'Unknown'} [🔍 Search on Spotify]({search_url})")


    # ===================== 5. Mood Clusters (PCA + KMeans) =====================
        st.subheader("🧠 Mood Clusters Explained (PCA + KMeans)")
        st.markdown("Each dot is a song. We've grouped similar moods using AI. Colors = Vibes! 🎨")

        X_cluster = cleaned_data[['valence', 'energy', 'danceability', 'acousticness']].dropna()
        if X_cluster.empty:
            st.warning("Not enough data to cluster moods. Try selecting more genres or resetting filters.")
        else:
            filtered_data = cleaned_data.loc[X_cluster.index].copy()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)

            # Optional fix before clustering
            X_scaled = np.clip(X_scaled, -5, 5)

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

        # ===================== 6. Mood Map (Valence vs Energy) =====================
        st.subheader("🎨 Mood Map: Valence vs Energy by Mood (Interactive)")
        st.markdown("This chart maps songs by **happiness (valence)** vs **intensity (energy)**. Each dot is a song, color = mood 🎨")

        try:
            plot_data = filtered_data.sample(n=1000, random_state=42) if len(filtered_data) > 1000 else filtered_data
            plot_data = cleaned_data[
                (cleaned_data['valence'] > 0) &
                (cleaned_data['energy'] > 0)
            ].copy()

            if len(plot_data) > 1000:
                plot_data = plot_data.sample(n=1000, random_state=42)
            fig_mood_map = px.scatter(
                data_frame=plot_data,
                x="valence",
                y="energy",
                color="Mood",
                hover_data=["artist_name", "track_name"],
                title="🎨 Mood Map: Valence vs Energy by Mood",
            )
            st.plotly_chart(fig_mood_map, use_container_width=True)
        except Exception as e:
            st.warning("Could not load Plotly chart. Showing fallback.")
            try:
                fig_fallback, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=filtered_data, x="valence", y="energy", hue="Mood", alpha=0.6, ax=ax, legend=False)
                ax.set_title("Mood Map")
                ax.set_xlabel("Valence")
                ax.set_ylabel("Energy")
                st.pyplot(fig_fallback)
            except Exception as fallback_error:
                st.error(f"Both Plotly and Matplotlib failed. Error: {fallback_error}")

    # ===================== 7. Popularity Prediction Sliders =====================
    st.subheader("🎯 Popularity Prediction Demo")
    if model is not None:
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

    # ===================== 8. Top 20 Songs of 2024 =====================
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

    # ===================== 9. Surprise Me Feature =====================
    if st.button("🎲 Surprise Me with a Track!"):
        surprise = data_main.sample(1).iloc[0]
        st.info(f"🎵 Track: **{surprise['track_name']}** by *{surprise['artist_name']}* | Genre: {surprise['genres']} | Popularity: {surprise['popularity']}")
        if surprise['popularity'] > 80:
            st.balloons()
        elif surprise['popularity'] < 30:
            st.snow()

    # ===================== 10. Theme Configuration Tip =====================
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


# ===================== 📊 Data Insights Tab =====================
with analysis_tab:
    st.title("📊 Technical Data Analysis")
    st.markdown("Explore deeper audio feature relationships, artists, cultural patterns, and playlist characteristics.")

    # === 1. Correlation Heatmap ===
    st.subheader("🔗 Audio Feature Correlation")
    numeric_features = ['danceability', 'energy', 'valence', 'speechiness', 'acousticness', 'liveness', 'instrumentalness', 'loudness', 'tempo']
    corr_matrix = data_main[numeric_features].dropna().corr()
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig_corr)

    # === 2. Feature Distributions ===
    st.subheader("🎼 Feature Distributions by Language")
    selected_feature = st.selectbox("Choose an audio feature to explore:", numeric_features)
    fig_dist = px.violin(data_main, y=selected_feature, x='language', box=True, points="all", title=f"Distribution of {selected_feature} by Language")
    st.plotly_chart(fig_dist, use_container_width=True)

    # === 3. Top Artists and Albums ===
    st.subheader("🎤 Top Artists by Language")
    lang_filter = st.selectbox("Choose language:", sorted(data_main['language'].dropna().unique()))
    top_artists = data_main[data_main['language'] == lang_filter]['artist_name'].value_counts().head(10).reset_index()
    top_artists.columns = ['Artist', 'Track Count']
    fig_artists = px.bar(top_artists, x='Track Count', y='Artist', orientation='h', title=f"Top 10 Artists in {lang_filter}")
    st.plotly_chart(fig_artists, use_container_width=True)

    if 'album_name' in data_main.columns:
        st.subheader("💿 Top Albums by Language")
        top_albums = data_main[data_main['language'] == lang_filter]['album_name'].value_counts().head(10).reset_index()
        top_albums.columns = ['Album', 'Track Count']
        fig_albums = px.bar(top_albums, x='Track Count', y='Album', orientation='h', title=f"Top 10 Albums in {lang_filter}")
        st.plotly_chart(fig_albums, use_container_width=True)

    # === 4. Language Comparisons ===
    st.subheader("🌍 Compare Audio Features Between Languages")
    compare_feature = st.selectbox("Choose a feature to compare:", numeric_features, key="langcompare")
    compare_data = data_main[data_main[compare_feature] > 0].dropna(subset=['language'])
    fig_compare = px.box(compare_data, x='language', y=compare_feature, points="all", title=f"{compare_feature} Comparison by Language")
    st.plotly_chart(fig_compare, use_container_width=True)

    # === 5. Lyrics Sentiment Analysis (Optional Placeholder) ===
    st.subheader("🧠 Lyrics Sentiment Analysis (Optional)")
    st.markdown("If lyrics data is added in future, we can extract sentiment scores and compare them by genre, artist, or language.")
    st.info("📌 Currently not implemented — you can scrape lyrics from Genius or use NLP APIs like TextBlob or VADER.")

    # (Add more technical insights here later: artist stats, comparisons, etc.)



