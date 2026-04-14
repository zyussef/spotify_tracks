"""
Spotify Analysis — Interactive Dashboard
Run locally:  streamlit run app.py
Hosted on:    GitHub Pages via stlite (docs/index.html)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPOTIFY_GREEN  = "#1DB954"
SPOTIFY_BLACK  = "#191414"
SPOTIFY_GRAY   = "#535353"
ACCENT_PINK    = "#FF6B9D"
ACCENT_BLUE    = "#4A90D9"
ACCENT_ORANGE  = "#FF8C42"
ACCENT_YELLOW  = "#FFD700"

AUDIO_FEATURES = [
    "danceability", "energy", "valence",
    "acousticness", "instrumentalness",
    "speechiness", "liveness",
]

# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    # Works locally; falls back to GitHub raw URL when running via stlite/GitHub Pages
    try:
        df = pd.read_csv("datasets/spotify_tracks_cleaned.csv")
    except Exception:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/"
            "zyussef/spotify_tracks/"
            "main/datasets/spotify_tracks_cleaned.csv"
        )

    df_unique = df.drop_duplicates("track_id").reset_index(drop=True)
    df_unique["dur_min"] = df_unique["duration_ms"] / 60_000
    return df, df_unique


df, df_unique = load_data()

# Precompute genre aggregates (cached once)
@st.cache_data
def genre_stats(df):
    return (
        df.groupby("track_genre")
        .agg(
            avg_pop       = ("popularity",    "mean"),
            avg_valence   = ("valence",       "mean"),
            avg_energy    = ("energy",        "mean"),
            avg_dance     = ("danceability",  "mean"),
            avg_acoustic  = ("acousticness",  "mean"),
            avg_tempo     = ("tempo",         "mean"),
            tracks        = ("track_name",    "count"),
        )
        .reset_index()
    )

genre_df = genre_stats(df)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def ml_placeholder(title: str, icon: str, description: str, features: list, color: str):
    """Reusable 'coming soon' card for Phase 4 ML pages."""
    bg = "#0d1f0d" if color == SPOTIFY_GREEN else "#1a1a2e"
    st.markdown(
        f"""
        <div style='border:2px dashed {color};border-radius:16px;
                    padding:32px;text-align:center;margin-bottom:24px;background:{bg}'>
            <p style='font-size:48px;margin:0'>{icon}</p>
            <h2 style='color:{color};margin:12px 0 8px'>{title}</h2>
            <p style='color:#aaa;max-width:600px;margin:0 auto'>{description}</p>
            <span style='display:inline-block;margin-top:16px;padding:6px 16px;
                         border-radius:20px;background:{color}22;color:{color};
                         font-size:12px;font-weight:bold'>PHASE 4 — IN DEVELOPMENT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Planned features")
    for f in features:
        st.markdown(f"- {f}")


def radar_chart(values: list, labels: list, title: str, color: str = SPOTIFY_GREEN):
    """Return a Plotly polar/radar figure for one genre."""
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    fig = go.Figure(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=color + "44",
            line=dict(color=color, width=2),
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor=SPOTIFY_BLACK,
            radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#333"),
            angularaxis=dict(gridcolor="#333", color="white"),
        ),
        paper_bgcolor=SPOTIFY_BLACK,
        plot_bgcolor=SPOTIFY_BLACK,
        title=dict(text=title, font=dict(color="white", size=13)),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )
    return fig


def dark_layout(fig, title="", height=420):
    fig.update_layout(
        paper_bgcolor=SPOTIFY_BLACK,
        plot_bgcolor=SPOTIFY_BLACK,
        font=dict(color="white"),
        title=dict(text=title, font=dict(color="white", size=14)),
        height=height,
        xaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#444"),
        yaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#444"),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown(
    f"<h2 style='color:{SPOTIFY_GREEN};margin-bottom:0'>🎵 Spotify Analysis</h2>",
    unsafe_allow_html=True,
)
st.sidebar.caption("114k tracks · 114 genres · 31k artists")
st.sidebar.divider()

ALL_PAGES = [
    "📊  Overview",
    "🎸  Genre Explorer",
    "🔍  Track Finder",
    "🔬  Audio Deep Dive",
    "─────────────────────",
    "🤖  Genre Classifier",
    "🧩  Sound Clusters",
    "💡  Recommendation Engine",
]

# Disable the separator line from being selectable
page = st.sidebar.radio(
    "nav", ALL_PAGES,
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption("Phase 4 ML pages are in development.")


# ─────────────────────────────────────────────────────────────
# Page routing
# ─────────────────────────────────────────────────────────────

# ── 1. OVERVIEW ──────────────────────────────────────────────
if page == "─────────────────────":
    st.info("Select a page from the sidebar.")

elif page == "📊  Overview":
    st.title("What Makes a Hit?")
    st.caption("Exploring 114,000 Spotify tracks across 114 genres")
    st.divider()

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tracks",   f"{len(df_unique):,}")
    c2.metric("Genres",   df["track_genre"].nunique())
    c3.metric("Artists",  f"{df['artists'].nunique():,}")
    c4.metric("Median Popularity", int(df_unique["popularity"].median()))
    c5.metric("Viral Tracks (≥76)", f"{(df_unique['popularity'] >= 76).sum():,}")
    st.divider()

    col_left, col_right = st.columns([3, 2])

    # Popularity histogram
    with col_left:
        fig = px.histogram(
            df_unique, x="popularity", nbins=50,
            color_discrete_sequence=[SPOTIFY_GREEN],
            labels={"popularity": "Popularity Score", "count": "Tracks"},
        )
        fig.add_vline(x=df_unique["popularity"].median(), line_dash="dash",
                      line_color=ACCENT_PINK,
                      annotation_text=f"Median = {int(df_unique['popularity'].median())}",
                      annotation_font_color=ACCENT_PINK)
        fig.add_vline(x=76, line_dash="dot", line_color=ACCENT_ORANGE,
                      annotation_text="Viral threshold",
                      annotation_font_color=ACCENT_ORANGE)
        dark_layout(fig, "Popularity Distribution", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Popularity tier donut
    with col_right:
        bins   = [-1, 25, 50, 75, 100]
        labels = ["Low (0–25)", "Medium (26–50)", "High (51–75)", "Viral (76–100)"]
        tier   = pd.cut(df_unique["popularity"], bins=bins, labels=labels).value_counts().sort_index()
        fig2   = go.Figure(go.Pie(
            labels=tier.index, values=tier.values,
            hole=0.55,
            marker_colors=[SPOTIFY_GRAY, "#7a7a7a", ACCENT_BLUE, SPOTIFY_GREEN],
            textfont=dict(color="white"),
        ))
        fig2.update_layout(
            paper_bgcolor=SPOTIFY_BLACK,
            font=dict(color="white"),
            title=dict(text="Popularity Tiers", font=dict(color="white", size=14)),
            legend=dict(font=dict(color="white")),
            height=380,
            margin=dict(t=50, b=10),
            annotations=[dict(text=f"Only<br><b style='font-size:18px'>2.1%</b><br>go viral",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=12, color=SPOTIFY_GREEN))]
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("🏆 Top 20 Most Popular Tracks")
    top20 = (df_unique.nlargest(20, "popularity")
             [["track_name", "artists", "popularity", "track_genre", "dur_min"]]
             .reset_index(drop=True))
    top20.index += 1
    top20["dur_min"] = top20["dur_min"].round(2).astype(str) + " min"
    top20.columns = ["Track", "Artist(s)", "Popularity", "Genre", "Duration"]
    st.dataframe(top20, use_container_width=True, height=500)


# ── 2. GENRE EXPLORER ────────────────────────────────────────
elif page == "🎸  Genre Explorer":
    st.title("Genre Explorer")
    st.divider()

    # Mood map
    st.subheader("Genre Mood Map — Valence vs Energy")
    st.caption("Each dot is a genre. X = musical positivity (valence), Y = intensity (energy).")

    fig_mood = px.scatter(
        genre_df, x="avg_valence", y="avg_energy",
        color="avg_pop", size="tracks",
        hover_name="track_genre",
        hover_data={
            "avg_pop": ":.1f", "avg_valence": ":.3f",
            "avg_energy": ":.3f", "tracks": True,
        },
        color_continuous_scale="RdYlGn",
        labels={"avg_valence": "Valence (Positivity →)",
                "avg_energy":  "Energy (Intensity →)",
                "avg_pop":     "Avg Popularity"},
        size_max=20,
    )
    # Quadrant lines
    for x_val in [0.5]:
        fig_mood.add_vline(x=x_val, line_dash="dot", line_color="#444")
    fig_mood.add_hline(y=0.5, line_dash="dot", line_color="#444")

    for text, ax, ay, x, y in [
        ("Happy + Energetic", 30, -20, 0.82, 0.90),
        ("Angry + Intense",   30, -20, 0.18, 0.90),
        ("Sad + Calm",        30,  20, 0.18, 0.12),
        ("Happy + Relaxed",   30,  20, 0.82, 0.12),
    ]:
        fig_mood.add_annotation(x=x, y=y, text=text, showarrow=False,
                                font=dict(size=9, color="#555"))

    dark_layout(fig_mood, height=480)
    fig_mood.update_coloraxes(colorbar=dict(tickfont=dict(color="white"),
                                            title=dict(font=dict(color="white"))))
    st.plotly_chart(fig_mood, use_container_width=True)

    st.divider()

    # Per-genre deep dive
    st.subheader("Genre Deep Dive")
    col_sel, col_sort = st.columns([3, 2])
    with col_sel:
        genre_list = sorted(df["track_genre"].unique().tolist())
        selected_genre = st.selectbox("Select a genre", genre_list, index=genre_list.index("pop"))
    with col_sort:
        sort_by = st.selectbox("Sort tracks by", ["popularity", "danceability", "energy", "valence"])

    gdf = df[df["track_genre"] == selected_genre]
    row = genre_df[genre_df["track_genre"] == selected_genre].iloc[0]

    col_radar, col_tracks = st.columns([2, 3])

    with col_radar:
        vals = [row[f"avg_{f}"] if f"avg_{f}" in row.index else gdf[f].mean()
                for f in AUDIO_FEATURES]
        # Manually build vals from gdf means
        vals = [gdf[f].mean() for f in AUDIO_FEATURES]
        st.plotly_chart(
            radar_chart(vals, AUDIO_FEATURES, f"{selected_genre.upper()} Sound Profile"),
            use_container_width=True
        )
        st.metric("Avg Popularity", f"{row['avg_pop']:.1f}")
        st.metric("Avg Tempo",      f"{gdf['tempo'].mean():.0f} BPM")
        st.metric("Tracks",         f"{len(gdf):,}")

    with col_tracks:
        st.markdown(f"**Top tracks in {selected_genre}**")
        top_genre = (gdf.drop_duplicates("track_id")
                       .nlargest(15, sort_by)
                       [["track_name", "artists", "popularity", "danceability", "energy", "valence"]]
                       .reset_index(drop=True))
        top_genre.index += 1
        top_genre.columns = ["Track", "Artist(s)", "Popularity", "Dance", "Energy", "Valence"]
        st.dataframe(top_genre, use_container_width=True, height=380)

    # Feature distribution for selected genre vs all
    st.divider()
    st.subheader(f"Audio Feature Distribution — {selected_genre} vs all genres")
    feature_pick = st.selectbox("Feature", AUDIO_FEATURES, index=0)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df_unique[feature_pick], name="All genres",
        nbinsx=40, opacity=0.5,
        marker_color=SPOTIFY_GRAY, histnorm="probability density"
    ))
    fig_dist.add_trace(go.Histogram(
        x=gdf[feature_pick], name=selected_genre,
        nbinsx=40, opacity=0.8,
        marker_color=SPOTIFY_GREEN, histnorm="probability density"
    ))
    fig_dist.update_layout(barmode="overlay", legend=dict(font=dict(color="white")))
    dark_layout(fig_dist, f"{feature_pick} — {selected_genre} vs all genres", height=350)
    st.plotly_chart(fig_dist, use_container_width=True)


# ── 3. TRACK FINDER ──────────────────────────────────────────
elif page == "🔍  Track Finder":
    st.title("Track Finder")
    st.caption("Search any track and explore its audio fingerprint + similar-sounding songs.")
    st.divider()

    query = st.text_input("Search by track name or artist", placeholder="e.g. Blinding Lights")

    if query:
        mask    = (df_unique["track_name"].str.contains(query, case=False, na=False) |
                   df_unique["artists"].str.contains(query, case=False, na=False))
        results = df_unique[mask][["track_name", "artists", "popularity", "track_genre"]].head(20)

        if results.empty:
            st.warning("No tracks found. Try a different search term.")
        else:
            st.markdown(f"**{len(results)} result(s)**")
            sel_idx = st.selectbox(
                "Select a track",
                results.index,
                format_func=lambda i: f"{results.loc[i,'track_name']} — {results.loc[i,'artists']}"
            )
            track = df_unique.loc[sel_idx]

            st.divider()
            col_info, col_radar = st.columns([2, 2])

            with col_info:
                st.markdown(f"### {track['track_name']}")
                st.caption(f"**Artist(s):** {track['artists']}")
                st.caption(f"**Album:** {track['album_name']}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Popularity",  int(track["popularity"]))
                m2.metric("Genre",       track["track_genre"])
                m3.metric("Duration",    f"{track['dur_min']:.2f} min")

                m4, m5, m6 = st.columns(3)
                m4.metric("Explicit",    "Yes" if track["explicit"] else "No")
                m5.metric("Tempo",       f"{track['tempo']:.0f} BPM")
                m6.metric("Key",         f"{track.get('key_name','')} {track.get('mode_name','')}")

                st.divider()
                st.markdown("**Audio features**")
                for feat in AUDIO_FEATURES:
                    val = float(track[feat])
                    st.progress(val, text=f"{feat.capitalize()}: {val:.3f}")

            with col_radar:
                vals = [float(track[f]) for f in AUDIO_FEATURES]
                fig_r = radar_chart(vals, AUDIO_FEATURES,
                                    track["track_name"][:40], SPOTIFY_GREEN)
                st.plotly_chart(fig_r, use_container_width=True)

            # Similar tracks by cosine distance on audio features
            st.divider()
            st.subheader("Similar-sounding tracks")
            st.caption("Ranked by cosine similarity across all 7 audio features.")

            feat_matrix = df_unique[AUDIO_FEATURES].values.astype(float)
            track_vec   = np.array([float(track[f]) for f in AUDIO_FEATURES])

            # Normalise
            feat_norms   = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
            feat_norms[feat_norms == 0] = 1
            feat_normed  = feat_matrix / feat_norms

            track_norm   = track_vec / (np.linalg.norm(track_vec) or 1)
            sims         = feat_normed @ track_norm

            # Exclude the track itself
            sims[sel_idx] = -1
            top_sim_idx  = np.argsort(sims)[::-1][:10]

            similar = df_unique.iloc[top_sim_idx][
                ["track_name", "artists", "track_genre", "popularity",
                 "danceability", "energy", "valence"]
            ].reset_index(drop=True)
            similar.index += 1
            similar.columns = ["Track", "Artist(s)", "Genre", "Popularity",
                                "Dance", "Energy", "Valence"]
            st.dataframe(similar, use_container_width=True, height=340)
    else:
        st.info("Type a track name or artist above to begin.")


# ── 4. AUDIO DEEP DIVE ───────────────────────────────────────
elif page == "🔬  Audio Deep Dive":
    st.title("Audio Deep Dive")
    st.divider()

    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    numeric_cols = AUDIO_FEATURES + ["loudness", "tempo", "popularity"]
    corr = df_unique[numeric_cols].corr().round(2)

    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(tickfont=dict(color="white"),
                      title=dict(text="r", font=dict(color="white"))),
    ))
    dark_layout(fig_heat, "Pearson Correlation — Audio Features", height=480)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # Scatter — user picks axes
    st.subheader("Feature Explorer")
    all_num = AUDIO_FEATURES + ["loudness", "tempo", "popularity", "dur_min"]
    c1, c2, c3 = st.columns(3)
    x_feat = c1.selectbox("X axis",    all_num, index=all_num.index("energy"))
    y_feat = c2.selectbox("Y axis",    all_num, index=all_num.index("loudness"))
    c_feat = c3.selectbox("Color by",  ["popularity", "track_genre"] + AUDIO_FEATURES,
                          index=0)

    genre_filter = st.multiselect(
        "Filter genres (leave empty for all)",
        sorted(df["track_genre"].unique()),
        default=[]
    )

    plot_df = df_unique.copy()
    if genre_filter:
        plot_df = plot_df[plot_df["track_genre"].isin(genre_filter)]

    sample_df = plot_df.sample(min(5000, len(plot_df)), random_state=42)

    if c_feat == "popularity":
        fig_sc = px.scatter(
            sample_df, x=x_feat, y=y_feat, color=c_feat,
            color_continuous_scale="RdYlGn",
            hover_data=["track_name", "artists", "track_genre"],
            opacity=0.5, size_max=6,
        )
    else:
        fig_sc = px.scatter(
            sample_df, x=x_feat, y=y_feat, color=c_feat,
            hover_data=["track_name", "artists", "track_genre"],
            opacity=0.5, size_max=6,
        )

    dark_layout(fig_sc, f"{x_feat} vs {y_feat}", height=480)
    if c_feat == "popularity":
        fig_sc.update_coloraxes(
            colorbar=dict(tickfont=dict(color="white"),
                          title=dict(font=dict(color="white")))
        )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Popularity prediction R² callout
    st.divider()
    st.subheader("How much do audio features predict popularity?")
    col_stat, col_explain = st.columns([1, 3])
    with col_stat:
        st.markdown(
            f"""
            <div style='background:{SPOTIFY_BLACK};border:2px solid {ACCENT_PINK};
                        border-radius:12px;padding:20px;text-align:center'>
                <p style='color:{SPOTIFY_GRAY};font-size:13px;margin:0'>R² (Linear Model)</p>
                <p style='color:{ACCENT_PINK};font-size:48px;font-weight:bold;margin:4px 0'>3.3%</p>
                <p style='color:{SPOTIFY_GRAY};font-size:12px;margin:0'>of variance explained</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_explain:
        st.markdown("""
        A linear regression trained on all 10 audio features explains only **3.3%** of the
        variance in popularity. The remaining **96.7%** comes from factors invisible in the audio data:

        - 🎯 **Playlist placement** — being added to a Spotify editorial playlist can add millions of streams overnight
        - 📣 **Marketing spend** — label promotion, social media campaigns, sync licensing
        - 🌟 **Artist fame** — brand recognition drives streams independent of audio quality
        - 🎵 **Release timing** — cultural moments, trends, TikTok virality
        - 📅 **Recency** — Spotify's algorithm weights recent streams heavily

        > The uncomfortable truth: you can make a perfect-sounding song and it will be invisible without distribution.
        """)


elif page == "🤖  Genre Classifier":
    st.title("Genre Classifier")
    st.divider()
    ml_placeholder(
        title="Predict Genre from Sound",
        icon="🤖",
        description=(
            "A Random Forest classifier trained on all 7 audio features. "
            "Given a set of audio characteristics, it predicts which of 114 genres "
            "the track most likely belongs to — and shows where it went wrong."
        ),
        features=[
            "Train/test split accuracy report across all 114 genres",
            "Confusion matrix heatmap — see which genres get confused most (pop vs indie-pop?)",
            "Feature importance chart — which audio feature matters most for genre ID?",
            "Interactive predictor: adjust feature sliders → see predicted genre in real time",
            "Genre separability analysis: which genres are acoustically distinct vs overlapping?",
        ],
        color=SPOTIFY_GREEN,
    )

    st.divider()
    st.subheader("Preview — What we already know")
    st.info(
        "From Phase 3, genre fingerprints show that genres **do** have distinct audio shapes. "
        "Death-metal (energy spike), hip-hop (speechiness spike), and classical (instrumentalness spike) "
        "will be trivial to classify. Pop, indie-pop, and alt-rock will blur together."
    )

    # Show genre separability teaser
    st.subheader("Genre variance — how 'pure' is each genre's sound?")
    genre_var = df.groupby("track_genre")[AUDIO_FEATURES].std().mean(axis=1).sort_values()
    fig_var = px.bar(
        x=genre_var.values[:20], y=genre_var.index[:20],
        orientation="h", color=genre_var.values[:20],
        color_continuous_scale=[[0, SPOTIFY_GREEN], [1, ACCENT_PINK]],
        labels={"x": "Avg std across audio features", "y": ""},
    )
    fig_var.update_layout(showlegend=False, coloraxis_showscale=False)
    dark_layout(fig_var, "Most consistent-sounding genres (low = more classifiable)", height=420)
    st.plotly_chart(fig_var, use_container_width=True)


elif page == "🧩  Sound Clusters":
    st.title("Sound Clusters")
    st.divider()
    ml_placeholder(
        title="Natural Sound Groups Across All Genres",
        icon="🧩",
        description=(
            "K-Means clustering on normalised audio features, visualised in 2D via PCA. "
            "Clusters represent natural 'mood groups' that cut across genre labels — "
            "a latin track and a pop track may cluster together if they sound similar."
        ),
        features=[
            "Elbow chart to determine optimal number of clusters (k)",
            "PCA scatter: 2D projection of all tracks, coloured by cluster vs by genre",
            "Cluster profile cards: radar chart showing the audio fingerprint of each cluster",
            "Cluster names inferred from dominant genres (e.g. 'High-energy Electric', 'Calm Acoustic')",
            "Filter by cluster to explore its tracks and genre composition",
            "Silhouette score to measure cluster quality",
        ],
        color=ACCENT_BLUE,
    )

    st.divider()
    st.subheader("Preview — Genre overlap suggests natural clusters exist")
    # Show PCA teaser with random projection
    st.info(
        "Phase 3 showed that acousticness ↔ energy (r = −0.73) and energy ↔ loudness (r = +0.76) "
        "create strong structure in the feature space. K-Means will find at least 4–6 meaningful "
        "clusters: Acoustic/Calm, Electric/High-energy, Dance/Happy, Ambient/Dark, Vocal, Instrumental."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig_ac = px.scatter(
            df_unique.sample(3000, random_state=42),
            x="acousticness", y="energy",
            color="track_genre", opacity=0.4,
            hover_data=["track_name", "artists"],
        )
        fig_ac.update_traces(marker=dict(size=4))
        fig_ac.update_layout(showlegend=False)
        dark_layout(fig_ac, "Acousticness vs Energy (structure for clustering)", height=380)
        st.plotly_chart(fig_ac, use_container_width=True)
    with col2:
        fig_dv = px.scatter(
            df_unique.sample(3000, random_state=42),
            x="danceability", y="valence",
            color="track_genre", opacity=0.4,
            hover_data=["track_name", "artists"],
        )
        fig_dv.update_traces(marker=dict(size=4))
        fig_dv.update_layout(showlegend=False)
        dark_layout(fig_dv, "Danceability vs Valence (mood axis)", height=380)
        st.plotly_chart(fig_dv, use_container_width=True)


elif page == "💡  Recommendation Engine":
    st.title("Recommendation Engine")
    st.divider()
    ml_placeholder(
        title="Find Your Next Favourite Track",
        icon="💡",
        description=(
            "A content-based recommendation system using cosine similarity on audio features. "
            "Phase 4 upgrades the basic Track Finder with genre-aware filtering, "
            "popularity weighting, and a hybrid model that combines audio similarity with "
            "collaborative filtering signals."
        ),
        features=[
            "Audio-only similarity: cosine distance across all 7 features (basic version in Track Finder now)",
            "Genre-aware filtering: option to stay within genre or explore cross-genre matches",
            "Popularity-weighted ranking: surface popular similar tracks above obscure ones",
            "Vibe slider: tune the recommendation toward 'more energetic', 'sadder', 'more danceable'",
            "Playlist generator: input N tracks → output a 20-track playlist with smooth transitions",
            "Evaluation: precision@k using implicit feedback from genre co-occurrence",
        ],
        color=ACCENT_ORANGE,
    )

    st.divider()
    st.info(
        "**Basic version available now in Track Finder (🔍).** "
        "Search any track and scroll to 'Similar-sounding tracks' — "
        "that's cosine similarity on 7 audio features, live."
    )

    st.subheader("Try the basic recommender")
    query2 = st.text_input("Quick search", placeholder="e.g. Blinding Lights")
    if query2:
        mask2 = (df_unique["track_name"].str.contains(query2, case=False, na=False) |
                 df_unique["artists"].str.contains(query2, case=False, na=False))
        results2 = df_unique[mask2].head(5)
        if not results2.empty:
            pick = results2.iloc[0]
            feat_matrix = df_unique[AUDIO_FEATURES].values.astype(float)
            pick_vec    = np.array([float(pick[f]) for f in AUDIO_FEATURES])
            norms       = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            sims = (feat_matrix / norms) @ (pick_vec / (np.linalg.norm(pick_vec) or 1))
            sims[results2.index[0]] = -1
            top_idx = np.argsort(sims)[::-1][:8]
            rec_df  = df_unique.iloc[top_idx][
                ["track_name", "artists", "track_genre", "popularity"]
            ].reset_index(drop=True)
            rec_df.index += 1
            st.markdown(f"**Because you like:** *{pick['track_name']}* — {pick['artists']}")
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.warning("No tracks found.")
