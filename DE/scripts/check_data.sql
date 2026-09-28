-- =============================================================
-- ChordSensePro DE — Data Health Check Queries
-- Run manually: docker exec chordsense_postgres psql -U chordsense -d chordsense -f /opt/airflow/scripts/check_data.sql
-- =============================================================

-- ═══ 1. crawl_queue: tổng quan trạng thái seed ═══
SELECT status, COUNT(*) FROM crawl_queue GROUP BY status;

-- Chi tiết failed
SELECT id, source_type, source_url, error_message 
FROM crawl_queue WHERE status = 'failed' LIMIT 10;

-- ═══ 2. song_analyses: kết quả DAG 2 ═══
SELECT status, COUNT(*) FROM song_analyses GROUP BY status;

-- Xem bài đã xử lý xong
SELECT id, song_title, detected_key, tempo_bpm, 
       jsonb_array_length(chord_timeline) AS num_chords
FROM song_analyses WHERE status = 'done' LIMIT 10;

-- Xem bài lỗi
SELECT id, source_url, error_message 
FROM song_analyses WHERE status = 'failed' LIMIT 10;

-- ═══ 3. chord_attempts + mastery ═══
SELECT COUNT(*) AS total_attempts FROM chord_attempts;
SELECT COUNT(*) AS total_mastery FROM user_chord_mastery;

-- ═══ Quick all-in-one ═══
SELECT 'crawl_queue' AS tbl, status, COUNT(*) FROM crawl_queue GROUP BY status
UNION ALL
SELECT 'song_analyses', status, COUNT(*) FROM song_analyses GROUP BY status;
