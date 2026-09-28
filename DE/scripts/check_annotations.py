"""Inspect JAAH annotation structure + check what AnnotationHandler gets."""
import json
from pathlib import Path

# 1. Check JAAH annotation file
f = Path("/data/datasets/jaah/JAAH-v0.1/MTG-JAAH-7686b91/annotations/giant_steps.json")
d = json.loads(f.read_text())
print("=== JAAH Annotation: giant_steps.json ===")
print(f"Keys: {list(d.keys())}")
print(f"Artist: {d.get('artist')}")
print(f"Title: {d.get('title')}")

anns = d.get("annotations", [])
print(f"Annotations count: {len(anns)}")
if anns:
    a = anns[0]
    print(f"Namespace: {a.get('namespace')}")
    data = a.get("data", [])
    print(f"Chord events: {len(data)}")
    for e in data[:8]:
        print(f"  t={e['time']:.2f}s  dur={e['duration']:.2f}s  chord={e['value']}")
    print(f"  ... ({len(data)} total)")

# 2. Check what parsers.py would return
print("\n=== JamsParser output ===")
# JAAH uses different format - it's not standard JAMS
# Let's check if namespace == 'chord'
for a in anns:
    print(f"  namespace: {a.get('namespace')}")

# 3. Check a MongoDB doc to see annotation_path
print("\n=== MongoDB: check annotation_path ===")
import asyncio
import sys
sys.path.insert(0, "/opt/airflow")
from src.shared.infrastructure.mongo.client import get_mongo_db

async def check_mongo():
    db = get_mongo_db()
    col = db["raw_audio_jobs"]
    # Find a JAAH job
    jaah = await col.find_one({"source_type": "jaah", "status": "done"})
    if jaah:
        print(f"  source_url: {jaah.get('source_url')}")
        print(f"  annotation_path: {jaah.get('annotation_path')}")
        print(f"  wav_path: {jaah.get('wav_path')}")
    else:
        print("  No done JAAH job found")
    
    # Find a Kaggle job
    kaggle = await col.find_one({"source_type": "local", "status": "done"})
    if kaggle:
        print(f"\n  Kaggle source_url: {kaggle.get('source_url')}")
        print(f"  Kaggle annotation_path: '{kaggle.get('annotation_path')}'")

asyncio.run(check_mongo())

# 4. Check crawl_queue annotation_url for JAAH
print("\n=== crawl_queue annotation_url ===")
import psycopg2
conn = psycopg2.connect(
    host="postgres", port=5432, dbname="chordsense",
    user="chordsense", password="chordsense_dev"
)
cur = conn.cursor()
cur.execute("SELECT source_url, annotation_url FROM crawl_queue WHERE source_type='jaah' AND annotation_url != '' LIMIT 3")
for row in cur.fetchall():
    print(f"  url: {row[0][:60]}...")
    print(f"  annotation: {row[1]}")
cur.execute("SELECT source_url, annotation_url FROM crawl_queue WHERE source_type='local' LIMIT 2")
for row in cur.fetchall():
    print(f"\n  Kaggle url: {row[0]}")
    print(f"  Kaggle annotation: '{row[1]}'")
conn.close()
