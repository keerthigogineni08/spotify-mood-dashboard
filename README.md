# 🎧 Spotify Mood Dashboard

An interactive data science app that visualizes song moods using Spotify's audio features, clustering, and machine learning.

🔗 **[Try the Live App](https://spotify-mood-dashboard-keerthi-gogineni.streamlit.app/)**

---

## 📊 Features

- 🎨 **Mood Map** – Visualize valence (happiness) vs. energy for thousands of tracks
- 🧭 **Radar Chart** – Explore the musical fingerprint of any song
- 🤖 **Mood Clusters** – Group songs using PCA + KMeans into moods like *Party Hype*, *Chill & Mellow*, etc.
- 📈 **Popularity Predictor** – ML model that predicts a song's popularity
- 🔥 **Top Hits 2024** – Visual overview of this year’s most streamed songs

---

## 🛠️ Tech Stack

- **Python** (Pandas, Seaborn, Scikit-learn, Matplotlib)
- **Streamlit** – for web app interactivity
- **Plotly** – for dynamic visualizations
- **Spotify datasets** – 1M+ songs + genre summaries

---

## 🧠 Behind the Project

I built this to combine my love for data, music, and storytelling in one place — the result is a dashboard that turns audio data into interactive emotional insights.

Whether you're analyzing vibes or exploring your own playlist, this tool helps you feel the music deeper.

---

## 💻 Run Locally

```bash
git clone https://github.com/keerthigogineni08/spotify-mood-dashboard.git
cd spotify-mood-dashboard
pip install -r requirements.txt
streamlit run spotify_mood_dashboard.py
