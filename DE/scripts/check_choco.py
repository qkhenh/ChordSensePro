"""Check ChoCo JAMS format — is it standard JAMS with namespace='chord'?"""
import json
from pathlib import Path

# Check one partition JAMS
f = Path("/data/datasets/choco/choco/partitions/biab-internet-corpus/choco/jams/biab-internet-corpus_0.jams")
d = json.loads(f.read_text())

print("=== ChoCo JAMS file ===")
print(f"Keys: {list(d.keys())}")

meta = d.get("file_metadata", {})
print(f"file_metadata keys: {list(meta.keys())}")
print(f"  title: {meta.get('title')}")
print(f"  identifiers: {meta.get('identifiers')}")
print(f"  duration: {meta.get('duration')}")

anns = d.get("annotations", [])
print(f"\nAnnotations count: {len(anns)}")
for i, a in enumerate(anns[:3]):
    ns = a.get("namespace")
    data = a.get("data", [])
    print(f"\n  [{i}] namespace={ns}, events={len(data)}")
    for e in data[:5]:
        print(f"    t={e.get('time')}, dur={e.get('duration')}, val={e.get('value')}")

# Check how many partitions have YouTube IDs
import os
count = 0
yt_count = 0
for jams_path in Path("/data/datasets/choco/choco/partitions").rglob("*.jams"):
    count += 1
    if count > 200:
        break
    try:
        dd = json.loads(jams_path.read_text())
        ids = dd.get("file_metadata", {}).get("identifiers", {})
        yt = ids.get("youtube_id") or ids.get("youtube") or ids.get("id_youtube")
        if yt:
            yt_count += 1
    except:
        pass

print(f"\nChecked {count} ChoCo JAMS, {yt_count} have YouTube ID")
