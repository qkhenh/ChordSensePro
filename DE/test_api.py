import urllib.request, json, io, zipfile

print("=== Try JAAH GitHub paths ===")
paths_to_try = [
    "https://raw.githubusercontent.com/MTG/JAAH/master/data/Alone_Together.json",
    "https://raw.githubusercontent.com/MTG/JAAH/master/annotations/Alone_Together.json",
    "https://raw.githubusercontent.com/MTG/JAAH/master/JAAH/Alone_Together.json",
]
for path in paths_to_try:
    try:
        r = urllib.request.urlopen(path, timeout=10)
        content = json.loads(r.read())
        print(f"OK: {path}")
        print("Keys:", list(content.keys())[:10])
        # Look for audio source
        for key in ["youtube_url", "youtubeURL", "youtube", "audio_url", "url"]:
            if key in content:
                print(f"  Audio field '{key}':", content[key])
        break
    except Exception as e:
        print(f"FAIL {path}: {e}")

print("\n=== Try JAAH zip sampling (first 64KB to get file list) ===")
try:
    # Fetch partial zip to read central directory
    req = urllib.request.Request(
        "https://zenodo.org/api/records/1290737/files/MTG/JAAH-v0.1.zip/content",
        headers={"Range": "bytes=-65536"}  # last 64KB contains ZIP central directory
    )
    r = urllib.request.urlopen(req, timeout=30)
    data = r.read()
    print(f"Downloaded {len(data)} bytes from zip tail")
    # Try to find JSON filenames in raw bytes
    text = data.decode("utf-8", errors="ignore")
    json_files = [line.strip() for line in text.split("\x00") if ".json" in line and len(line) < 100]
    print("JSON files found:", json_files[:10])
except Exception as e:
    print("Zip sample ERROR:", e)

print("\n=== Try ChoCo partition files ===")
paths_to_try_choco = [
    "https://raw.githubusercontent.com/smashub/choco/main/choco/choco.csv",
    "https://raw.githubusercontent.com/smashub/choco/main/choco/choco-metadata.json",
    "https://raw.githubusercontent.com/smashub/choco/main/choco/partitions/rock/4ad5b9df-bca5-4a05-9e89-d5c5e9c31e58/ann_audio_chord.jams",
    "https://raw.githubusercontent.com/smashub/choco/main/choco/partitions/rock/meta.json",
]
for path in paths_to_try_choco:
    try:
        r = urllib.request.urlopen(path, timeout=10)
        content = r.read(200)
        print(f"OK: {path}")
        print("  Content:", content[:100])
        break
    except Exception as e:
        print(f"FAIL: {e}")
