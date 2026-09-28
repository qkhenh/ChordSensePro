"""Check JAAH beats structure — do they have timestamps?"""
import json
from pathlib import Path

f = Path("/data/datasets/jaah/JAAH-v0.1/MTG-JAAH-7686b91/annotations/giant_steps.json")
d = json.loads(f.read_text())

part = d["parts"][0]
print(f"Part name: {part['name']}")
print(f"Beats type: {type(part['beats'])}")
beats = part["beats"]
print(f"Beats count: {len(beats)}")
print(f"First 10 beats:")
for b in beats[:10]:
    print(f"  {b}")

print(f"\nChords type: {type(part['chords'])}")
print(f"Chords count: {len(part['chords'])}")
print(f"First chord line: {part['chords'][0][:100]}")

print(f"\nDuration: {d.get('duration')}")
print(f"Metre: {d.get('metre')}")
