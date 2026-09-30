import zlib, urllib.request

def encode_plantuml(text):
    table = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_'
    compressed = zlib.compress(text.encode('utf-8'))[2:-4]
    res = []
    for i in range(0, len(compressed), 3):
        b1 = compressed[i]
        b2 = compressed[i+1] if i+1 < len(compressed) else 0
        b3 = compressed[i+2] if i+2 < len(compressed) else 0
        res.append(table[b1 >> 2])
        res.append(table[((b1 & 0x3) << 4) | (b2 >> 4)])
        res.append(table[((b2 & 0xF) << 2) | (b3 >> 6)])
        res.append(table[b3 & 0x3F])
    return ''.join(res)

# Test 1: Condition diamond and direct merge without extra diamond
act_test = '''@startuml
skinparam conditionStyle diamond
skinparam activity {
    BackgroundColor #74C0FC
    BorderColor #1C7ED6
    FontColor #000000
    FontSize 12
    ArrowColor #2B2D42
    DiamondBackgroundColor #E7F5FF
    DiamondBorderColor #1C7ED6
}
skinparam partition {
    BorderColor #495057
    FontColor #212529
    FontSize 13
    BackgroundColor #FFFFFF
}

|Musician (User)|
start
:Upload audio file or enter YouTube URL;

|Frontend Client|
:Validate audio source format & duration;
if (Is audio source valid?) then (Yes)

|Backend Server|
:Generate Job ID and enqueue processing task;

|AI Pipeline Worker|
:Separate instrument stems using Demucs;
:Track tempo and downbeat positions using Madmom;
:Recognize chord progression sequence using MERT;
:Assemble event-based timeline JSON;

|Frontend Client|
:Fetch metadata and chord event timeline;
:Harmonic Engine computes Roman numerals & scale modes;
:Render interactive Smart Grid synced with Web Audio;

|Musician (User)|
:Listen and audit detected chord progression;
if (Chord correction needed?) then (Yes)
    :Open Edit Marker on target bar;
    :Input modified root, bass, or chord quality;
    |Frontend Client|
    :Recalculate Roman degree and scale mode instantly;
    :Flag chord event as user-edited;
    |Musician (User)|
    :Practice with A/B loop, dynamic metronome, and speed shift;
else (No)
    |Musician (User)|
    :Practice with A/B loop, dynamic metronome, and speed shift;
endif

|Musician (User)|
:Save custom chart to personal playlist;

|Backend Server|
:Persist custom chart into PostgreSQL JSONB;

|Frontend Client|
:Display sync confirmation toast;
stop

|Frontend Client|
else (No)
    :Display invalid audio source error;
    stop
endif
@enduml'''

enc = encode_plantuml(act_test)
req = urllib.request.Request(f'https://www.plantuml.com/plantuml/png/{enc}', headers={'User-Agent': 'Mozilla/5.0'})
data = urllib.request.urlopen(req).read()
with open('diagrams/test_act_diamond.png', 'wb') as f:
    f.write(data)
print('Done test_act_diamond')
