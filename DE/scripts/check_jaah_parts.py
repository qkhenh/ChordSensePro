"""Check JAAH annotation 'parts' structure — the actual chord data."""
import json
from pathlib import Path

f = Path("/data/datasets/jaah/JAAH-v0.1/MTG-JAAH-7686b91/annotations/giant_steps.json")
d = json.loads(f.read_text())

print("=== JAAH 'parts' structure ===")
parts = d.get("parts", [])
print(f"Parts count: {len(parts)}")

for i, part in enumerate(parts[:3]):
    print(f"\n--- Part {i} ---")
    print(f"  Keys: {list(part.keys())}")
    print(f"  Name: {part.get('name', 'N/A')}")
    chords = part.get("chords", [])
    print(f"  Chords count: {len(chords)}")
    for c in chords[:5]:
        print(f"    {c}")

# Check second file
f2 = Path("/data/datasets/jaah/JAAH-v0.1/MTG-JAAH-7686b91/annotations/all_alone.json")
d2 = json.loads(f2.read_text())
print(f"\n=== all_alone.json ===")
print(f"Keys: {list(d2.keys())}")
parts2 = d2.get("parts", [])
print(f"Parts: {len(parts2)}")
if parts2:
    p = parts2[0]
    print(f"  Part keys: {list(p.keys())}")
    chords = p.get("chords", [])
    if chords:
        for c in chords[:5]:
            print(f"    {c}")
