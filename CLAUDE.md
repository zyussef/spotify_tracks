# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses a local conda environment at `.conda/`. Always use it:

```bash
.conda/bin/jupyter notebook          # launch Jupyter
.conda/bin/python script.py          # run a script
.conda/bin/pip install <package>     # install packages
```

Dependencies are tracked in `requirements.txt`: `pandas`, `matplotlib`, `seaborn`, `numpy`, `ipykernel`, `sqlalchemy`.

## Project Overview

Exploratory data analysis of ~114,000 Spotify tracks across 114 genres. The dataset contains track metadata and Spotify's audio feature scores (danceability, energy, valence, acousticness, etc.) plus popularity scores.

**Analysis phases:**
1. **`01_data_exploration.ipynb`** — Data profiling, cleaning pipeline, and export. Produces `datasets/spotify_tracks_cleaned.csv` and `datasets/dataset_raw_backup.csv`.
2. **`01_spotify_analysis.ipynb`** — Earlier exploratory notebook with scatter/histogram plots exploring energy, loudness, and popularity distributions.
3. **`sql_queries.sql`** — SQL queries designed to run against a PostgreSQL table called `tracks` (loaded via SQLAlchemy in the notebook).

## Dataset Structure

**Primary working dataset:** `datasets/dataset.csv` (raw) → `datasets/spotify_tracks_cleaned.csv` (cleaned)

Key columns after cleaning:
- `track_id`, `track_name`, `artists`, `album_name`, `track_genre`
- `popularity` (0–100; 0 means negligible streams, not an error — 14% of tracks)
- `duration_ms`, `explicit`
- Audio features (all 0–1 unless noted): `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`
- `loudness` (dB, negative), `tempo` (BPM), `key`, `mode`, `time_signature`

**Cleaning decisions made in `01_data_exploration.ipynb`:**
- Dropped `Unnamed: 0` (artifact index column from original CSV export)
- Removed 3 rows with null `artists`/`album_name`/`track_name`
- Kept duplicate `track_id` rows (same track appearing in multiple genres — intentional dataset design, 24,259 duplicates)

## PostgreSQL Integration

The notebook loads the cleaned data into a local PostgreSQL database:
```python
from sqlalchemy import create_engine
engine = create_engine("postgresql://localhost/spotify_db")
df_clean.to_sql("tracks", engine, if_exists="append", index=False)
```

`sql_queries.sql` contains analysis queries (genre counts, popularity tiers, vibe scores, window functions) meant to be run against this `tracks` table.

## Key Analysis Findings (from notebooks)

- Energy ↔ Loudness: strong positive correlation (r ≈ 0.76)
- Popularity is heavily right-skewed — hits are rare
- Dataset is balanced: 114 genres × ~1,000 tracks each
- Very short tracks (< 30s): 17 tracks — may skew duration analysis
