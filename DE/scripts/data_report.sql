-- =============================================================
-- ChordSensePro DE — Training data report (queue, segments, label classes)
-- Run (PowerShell, from repo root):
--   Get-Content DE\scripts\data_report.sql -Raw | docker exec -i chordsense_postgres psql -U chordsense -d chordsense
-- Read-only. Class names follow shared/domain/value_objects.py (enum definition order).
-- =============================================================

\echo '== crawl_queue: status per source =='
SELECT source_type, status, count(*) FROM crawl_queue GROUP BY 1,2 ORDER BY 1,2;

\echo '== crawl_queue: failed, top errors =='
SELECT source_type, left(coalesce(error_message,''), 70) AS err, count(*)
FROM crawl_queue WHERE status = 'failed' GROUP BY 1,2 ORDER BY 3 DESC LIMIT 25;

\echo '== samples per source / split =='
SELECT source_type, count(DISTINCT song_id) AS songs, count(*) AS segs,
       sum((split='train')::int) AS train, sum((split='val')::int) AS val, sum((split='test')::int) AS test
FROM training_samples GROUP BY 1 ORDER BY 3 DESC;

\echo '== label_status =='
SELECT label_status, count(*) FROM training_samples GROUP BY 1 ORDER BY 2 DESC;

\echo '== quality (ok + partial) =='
SELECT t.quality_id, q.name, count(*)
FROM training_samples t
LEFT JOIN (VALUES (0,'major'),(1,'minor'),(2,'dominant'),(3,'dim'),(4,'sus2'),(5,'sus4')) AS q(id,name) ON q.id = t.quality_id
WHERE t.label_status IN ('ok','partial') GROUP BY 1,2 ORDER BY 1;

\echo '== extension (ok + partial); dim7 = quality dim + ext dom7 =='
SELECT t.ext_id, e.name, count(*)
FROM training_samples t
LEFT JOIN (VALUES (0,'none'),(1,'b5'),(2,'#5'),(3,'maj7'),(4,'min7'),(5,'dom7'),(6,'add9'),(7,'b9'),
                  (8,'9'),(9,'#9'),(10,'11'),(11,'#11'),(12,'13'),(13,'b13'),(14,'alt')) AS e(id,name) ON e.id = t.ext_id
WHERE t.label_status IN ('ok','partial') GROUP BY 1,2 ORDER BY 1;

\echo '== root (ok + partial) =='
SELECT t.root_id, r.name, count(*)
FROM training_samples t
LEFT JOIN (VALUES (0,'C'),(1,'C#'),(2,'D'),(3,'D#'),(4,'E'),(5,'F'),(6,'F#'),(7,'G'),(8,'G#'),(9,'A'),(10,'A#'),(11,'B')) AS r(id,name)
       ON r.id = t.root_id
WHERE t.label_status IN ('ok','partial') GROUP BY 1,2 ORDER BY 1;

\echo '== rare classes per source =='
SELECT source_type,
  sum((quality_id=4)::int) AS sus2, sum((quality_id=5)::int) AS sus4, sum((quality_id=3)::int) AS dim, sum((ext_id=2)::int) AS aug,
  sum((ext_id=6)::int) AS add9, sum((ext_id IN (7,8,9))::int) AS nine,
  sum((ext_id IN (10,11))::int) AS eleven, sum((ext_id IN (12,13))::int) AS thirteen,
  sum((ext_id=14)::int) AS alt
FROM training_samples WHERE label_status IN ('ok','partial') GROUP BY 1 ORDER BY 1;

\echo '== top unmapped labels =='
SELECT label_raw, count(*) FROM training_samples WHERE label_status = 'unmapped' GROUP BY 1 ORDER BY 2 DESC LIMIT 15;

\echo '== throughput: segments per hour (UTC) =='
SELECT date_trunc('hour', created_at) AS h, count(DISTINCT song_id) AS songs, count(*) AS segs
FROM training_samples GROUP BY 1 ORDER BY 1 DESC LIMIT 14;
