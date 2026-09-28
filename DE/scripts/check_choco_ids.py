"""Check ChoCo partitions for YouTube IDs and chord_harte namespace."""
import json
from pathlib import Path

parts_root = Path("/data/datasets/choco/choco/partitions")
dirs = sorted([d.name for d in parts_root.iterdir() if d.is_dir()])
print(f"Partition dirs: {len(dirs)}")
for d in dirs:
    print(f"  {d}")

# Sample JAMS from each partition
total = 0
yt_total = 0
has_chord = 0
for d in dirs:
    jams_dir = parts_root / d / "choco" / "jams"
    if not jams_dir.exists():
        continue
    for f in list(jams_dir.glob("*.jams"))[:5]:
        total += 1
        try:
            dd = json.loads(f.read_text())
            ids = dd.get("file_metadata", {}).get("identifiers", {})
            yt = ids.get("youtube_id") or ids.get("youtube") or ids.get("id_youtube")
            if yt:
                yt_total += 1
            for a in dd.get("annotations", []):
                if "chord" in a.get("namespace", ""):
                    has_chord += 1
                    break
        except:
            pass

print(f"\nSampled {total} JAMS:")
print(f"  {yt_total} have YouTube ID")
print(f"  {has_chord} have chord annotation")

# Show identifier keys from different partitions
print("\n=== Sample identifiers ===")
for d in dirs[:5]:
    jams_dir = parts_root / d / "choco" / "jams"
    if not jams_dir.exists():
        continue
    sample = list(jams_dir.glob("*.jams"))[:1]
    for s in sample:
        dd = json.loads(s.read_text())
        ids = dd.get("file_metadata", {}).get("identifiers", {})
        namespaces = [a.get("namespace") for a in dd.get("annotations", [])]
        print(f"  [{d}] identifiers={ids}, namespaces={namespaces}")
