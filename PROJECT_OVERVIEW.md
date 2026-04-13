# Spotify Data Analysis — Full Project Overview

A roadmap for turning the current EDA into a production-quality, publicly shareable data project.

---

## Project Architecture

```
Raw Data (Kaggle CSV)
        │
        ▼
┌───────────────────┐
│  Phase 1: EDA     │  ← Done ✓ (01_data_exploration.ipynb)
│  Profile & Clean  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Phase 2: SQL     │  ← Done ✓ (sql_queries.sql + PostgreSQL)
│  Aggregate Queries│
└────────┬──────────┘
         │
         ▼
┌────────────────────────┐
│  Phase 3: Analysis     │  ← In progress (02_data_story.ipynb)
│  Correlations, Trends  │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Phase 4: ML Modeling  │  ← Next
│  Clustering, Prediction│
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Phase 5: Dashboard    │  ← Final
│  Interactive Web App   │
└────────────────────────┘
```

---

## Phase 3 — Deeper Analysis (Notebooks)

What's worth building beyond the current story:

### 3a. Genre Fingerprinting
Plot a radar chart (spider/polar) of the avg audio feature profile per genre.
Each genre gets a unique "sound shape" — death metal vs salsa vs ambient look completely different.
Library: `matplotlib` polar axes or `plotly` radar chart.

### 3b. Artist Consistency Analysis
For artists with 5+ tracks, measure the standard deviation of their audio features.
Low σ = formula artists (predictable sound). High σ = experimenters.
Interesting question: does consistency correlate with popularity?

### 3c. TikTok Proxy — Tempo & Danceability Interaction
Tracks under 3 min with high danceability + high tempo are the TikTok-era format.
Segment these and see if they skew younger genres and higher popularity post-2018.
(Dataset doesn't have release year, but genre can serve as a proxy.)

### 3d. Popularity Prediction — What We Can and Can't Explain
Build a simple linear regression:
`popularity ~ danceability + energy + valence + tempo + explicit + duration_ms`
Report R² — it'll be low (~0.05), which is itself the story: popularity is structurally unpredictable from sound alone.

---

## Phase 4 — Machine Learning

### 4a. Genre Classification
**Task:** Given audio features only, can a model predict `track_genre`?
**Model:** Random Forest or XGBoost classifier.
**Why it's interesting:** Measures how acoustically distinct genres are.
Some genres will be nearly indistinguishable (pop vs indie-pop); others will be trivial (death-metal vs sleep).

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df[['danceability','energy','valence','acousticness',
        'instrumentalness','speechiness','liveness','tempo','loudness']]
y = df['track_genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
```

Expected accuracy: ~40–55% across 114 classes. Chance = 0.9%.

### 4b. Track Clustering — Finding Natural "Sound Groups"
**Task:** K-Means or DBSCAN on audio features to find clusters that don't map to genre labels.
**Interesting outcome:** Clusters likely correspond to moods (upbeat/acoustic/dark/electronic)
that cut across genre boundaries.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=8, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
```

Plot the PCA projection colored by cluster vs colored by genre — the mismatch is revealing.

### 4c. Popularity Regression
**Task:** Predict popularity score (regression, not classification).
**Models to try:** Ridge regression baseline → Gradient Boosting → Neural Net.
**Key insight to surface:** Even the best model will have R² < 0.15. That's the point.
Frame it as: "We can explain 14% of why a song is popular. The other 86% is everything except the music."

### 4d. Recommendation Engine (Simple Version)
**Task:** Given a track the user likes, find the N most similar tracks by audio feature distance.
**Method:** Cosine similarity on normalized feature vectors.

```python
from sklearn.metrics.pairwise import cosine_similarity

features_matrix = StandardScaler().fit_transform(X)
similarity_matrix = cosine_similarity(features_matrix)

def recommend(track_name, n=5):
    idx = df[df['track_name'] == track_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return df.iloc[[i for i, _ in scores]][['track_name','artists','track_genre']]
```

This becomes the interactive centerpiece of the dashboard.

---

## Phase 5 — Dashboard / Web App

### Option A — Streamlit (Recommended for this project)

**Best fit for:** Data-heavy apps, rapid development, Python-native, no frontend skills needed.

```python
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Spotify Analysis", layout="wide")
df = pd.read_csv('datasets/spotify_tracks_cleaned.csv')

st.title("What Makes a Hit?")
genre = st.selectbox("Pick a genre", df['track_genre'].unique())
filtered = df[df['track_genre'] == genre]
fig = px.scatter(filtered, x='energy', y='danceability', color='popularity',
                 hover_data=['track_name','artists'], color_continuous_scale='Viridis')
st.plotly_chart(fig, use_container_width=True)
```

Key pages to build:
- **Overview** — summary stats, popularity distribution
- **Genre Explorer** — mood map, filter by genre, radar chart
- **Track Finder** — search a track → see its audio fingerprint + similar tracks
- **Predict My Hit** — sliders for audio features → model returns predicted popularity

### Option B — Dash by Plotly

**Best fit for:** More control over layout, closer to a proper web app, still Python-native.
More verbose than Streamlit but supports complex callback logic better.

### Option C — Observable / D3.js

**Best fit for:** If you want stunning, custom visualizations beyond what Python libraries offer.
Steep learning curve but the output is visually unmatched. Export data to JSON, build charts in JS.
Good for a portfolio piece where aesthetics matter as much as analysis.

---

## Hosting Options

### Free Tier

| Platform | Best For | How |
|----------|----------|-----|
| **Streamlit Community Cloud** | Streamlit apps | Connect GitHub repo → auto-deploys on push. Zero config. |
| **GitHub Pages** | Static sites (no Python backend) | Export charts as HTML with `plotly.write_html()`, serve as static page |
| **Render** | Streamlit / Dash / Flask | Free tier with spin-down on inactivity (~30s cold start) |
| **Hugging Face Spaces** | Streamlit or Gradio apps | Free GPU option available, great for ML-heavy apps |

### Paid / Production

| Platform | Best For | Notes |
|----------|----------|-------|
| **Railway** | Full-stack Python apps | $5/mo, no cold starts, PostgreSQL add-on available |
| **Fly.io** | Containerized apps (Docker) | Good free tier, scales well, more DevOps required |
| **Vercel** | Static exports or Next.js | Best for Observable/D3 static builds, not Python backends |
| **AWS / GCP** | Full production scale | Overkill for a portfolio project but worth knowing |

### Recommended Path for This Project

```
1. Build app in Streamlit
2. Push to GitHub (public repo)
3. Deploy via Streamlit Community Cloud (free, instant, shareable URL)
4. Add PostgreSQL via Railway ($5/mo) if you want live querying instead of CSV
```

Streamlit Community Cloud deployment is literally:
1. Go to share.streamlit.io
2. Connect your GitHub repo
3. Point it at `app.py`
4. Get a public URL: `https://your-name-spotify-analysis.streamlit.app`

---

## Suggested Final File Structure

```
spotify-analysis/
├── datasets/
│   ├── dataset.csv                   # raw
│   └── spotify_tracks_cleaned.csv    # cleaned
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_story.ipynb
│   ├── 03_feature_analysis.ipynb     # Phase 3
│   └── 04_modeling.ipynb             # Phase 4
├── models/
│   └── genre_classifier.pkl          # saved sklearn model
├── app.py                            # Streamlit dashboard
├── requirements.txt
└── README.md                         # with screenshots + live demo link
```

---

## Tech Stack Summary

| Layer | Tool |
|-------|------|
| Data wrangling | `pandas`, `numpy` |
| Visualization (notebooks) | `matplotlib`, `seaborn` |
| Visualization (dashboard) | `plotly` (interactive) |
| Machine learning | `scikit-learn` |
| Database | PostgreSQL + `sqlalchemy` |
| Dashboard framework | `streamlit` |
| Hosting | Streamlit Community Cloud |
