-- What genres have the most tracks?
SELECT track_genre, COUNT(*) AS track_count
FROM tracks
GROUP BY track_genre
ORDER BY track_count DESC
LIMIT 15;

-- Top 20 most popular tracks
SELECT track_name, artists, popularity, track_genre
FROM tracks
WHERE popularity IS NOT NULL
ORDER BY popularity DESC
LIMIT 20;

-- Explicit vs non-explicit: which is more popular on average?
SELECT explicit,
       COUNT(*) AS track_count,
       ROUND(AVG(popularity), 2) AS avg_popularity
FROM tracks
GROUP BY explicit;

-- Average "vibe profile" for the top 10 genres
SELECT track_genre,
       COUNT(*) AS tracks,
       ROUND(AVG(danceability), 3) AS avg_dance,
       ROUND(AVG(energy), 3) AS avg_energy,
       ROUND(AVG(valence), 3) AS avg_valence,
       ROUND(AVG(acousticness), 3) AS avg_acoustic,
       ROUND(AVG(tempo), 1) AS avg_tempo
FROM tracks
GROUP BY track_genre
ORDER BY tracks DESC
LIMIT 10;

-- Happiest vs saddest genres (valence = musical positivity)
SELECT track_genre,
       ROUND(AVG(valence), 3) AS avg_valence,
       COUNT(*) AS tracks
FROM tracks
GROUP BY track_genre
HAVING COUNT(*) > 100
ORDER BY avg_valence DESC;

-- Most energetic genres
SELECT track_genre,
       ROUND(AVG(energy), 3) AS avg_energy,
       ROUND(AVG(loudness), 2) AS avg_loudness
FROM tracks
GROUP BY track_genre
HAVING COUNT(*) > 100
ORDER BY avg_energy DESC
LIMIT 10;

-- Popularity buckets: how many tracks fall in each tier?
SELECT
    CASE
        WHEN popularity BETWEEN 0 AND 25 THEN '0-25 Low'
        WHEN popularity BETWEEN 26 AND 50 THEN '26-50 Medium'
        WHEN popularity BETWEEN 51 AND 75 THEN '51-75 High'
        WHEN popularity BETWEEN 76 AND 100 THEN '76-100 Viral'
    END AS popularity_tier,
    COUNT(*) AS track_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct
FROM tracks
GROUP BY popularity_tier
ORDER BY popularity_tier;

-- Duration analysis: are shorter tracks more popular?
SELECT
    CASE
        WHEN duration_ms < 120000 THEN 'Under 2 min'
        WHEN duration_ms < 180000 THEN '2-3 min'
        WHEN duration_ms < 240000 THEN '3-4 min'
        WHEN duration_ms < 300000 THEN '4-5 min'
        ELSE 'Over 5 min'
    END AS duration_bucket,
    COUNT(*) AS tracks,
    ROUND(AVG(popularity), 2) AS avg_popularity
FROM tracks
GROUP BY duration_bucket
ORDER BY avg_popularity DESC;

-- "Vibe score" — composite of danceability + energy + valence
-- Which genres are the most fun to listen to?
SELECT track_genre,
       ROUND(AVG(danceability + energy + valence) / 3, 3) AS vibe_score,
       COUNT(*) AS tracks
FROM tracks
GROUP BY track_genre
HAVING COUNT(*) > 100
ORDER BY vibe_score DESC
LIMIT 15;

-- Rank genres by average popularity
SELECT track_genre,
       ROUND(AVG(popularity), 2) AS avg_popularity,
       RANK() OVER(ORDER BY AVG(popularity) DESC) AS popularity_rank
FROM tracks
GROUP BY track_genre
LIMIT 20;

-- For each genre, find the most popular track
SELECT track_genre, track_name, artists, popularity
FROM (
    SELECT track_genre, track_name, artists, popularity,
           ROW_NUMBER() OVER(PARTITION BY track_genre ORDER BY popularity DESC) AS rn
    FROM tracks
) ranked
WHERE rn = 1
ORDER BY popularity DESC
LIMIT 20;

-- How does each track's popularity compare to its genre average?
SELECT track_name, artists, track_genre, popularity,
       ROUND(AVG(popularity) OVER(PARTITION BY track_genre), 2) AS genre_avg,
       popularity - ROUND(AVG(popularity) OVER(PARTITION BY track_genre), 2) AS above_avg
FROM tracks
ORDER BY above_avg DESC
LIMIT 20;

-- Energy vs acousticness by genre — are they inversely related?
SELECT track_genre,
       ROUND(AVG(energy), 3) AS avg_energy,
       ROUND(AVG(acousticness), 3) AS avg_acoustic
FROM tracks
GROUP BY track_genre
HAVING COUNT(*) > 100
ORDER BY avg_energy DESC;

-- Danceability vs popularity scatter data
SELECT danceability, popularity, track_genre
FROM tracks
WHERE track_genre IN ('pop', 'hip-hop', 'latin', 'rock', 'electronic');


