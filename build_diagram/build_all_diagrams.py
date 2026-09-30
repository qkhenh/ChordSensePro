import os
import zlib
import urllib.request
import re
import subprocess

os.makedirs('diagrams', exist_ok=True)

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

print("=== 1. GENERATING USE CASE DIAGRAM (UPDATED 10 UCs) ===")
usecase_puml = """@startuml
left to right direction
skinparam packageStyle rectangle
skinparam actor {
    BorderColor #E03131
    BackgroundColor #FFE3E3
    FontColor #212529
    FontSize 13
}
skinparam usecase {
    BackgroundColor #0CA678
    BorderColor #087F5B
    FontColor #FFFFFF
    FontSize 11
}
skinparam rectangle {
    BackgroundColor #F8F9FA
    BorderColor #ADB5BD
    FontColor #212529
}
skinparam package {
    BackgroundColor #FFFFFF
    BorderColor #CED4DA
    FontColor #495057
    FontSize 13
}
skinparam arrow {
    Color #868E96
    FontColor #495057
}

' -- Left Actors --
actor "Guest" as Guest
actor "Free User" as FreeUser
actor "Paid User (Pro)" as PaidUser
actor "Expert\\n(Music Theory Advisor)" as Expert
actor "Music Platform Manager\\n(Business Admin)" as Manager
actor "System Administrator\\n(IT Admin)" as ITAdmin

rectangle "ChordSense Pro System" {
    usecase "UC-00: Register" as UC00
    usecase "UC-01: Login" as UC01
    usecase "UC-02: Import and\\nAnalyze Music" as UC02
    usecase "UC-03: Connect MIDI\\nInstrument (HITL)" as UC03
    usecase "UC-04: A/B Loop and\\nPractice Tools" as UC04
    usecase "UC-05: Edit Marker\\n(Override AI Chord)" as UC05
    usecase "UC-06: Real-Time Practice\\nMode / Microphone (HITL)" as UC06
    usecase "UC-07: Manage Playlist\\nand Library" as UC07
    usecase "UC-08: Monitor\\nAI Quality" as UC08
    usecase "UC-09: Deploy / Roll Back\\nAI Model" as UC09
}

' -- Actor Connections --
Guest --> UC00

FreeUser --> UC01
FreeUser --> UC02
FreeUser --> UC05
FreeUser --> UC07

PaidUser --> UC01
PaidUser --> UC02
PaidUser --> UC03
PaidUser --> UC04
PaidUser --> UC05
PaidUser --> UC06
PaidUser --> UC07

Expert --> UC01

Manager --> UC01
Manager --> UC08

ITAdmin --> UC01
ITAdmin --> UC09

@enduml"""

with open('diagrams/usecase_diagram.puml', 'w', encoding='utf-8') as f:
    f.write(usecase_puml)

enc_uc = encode_plantuml(usecase_puml)
req_uc_svg = urllib.request.Request(f'https://www.plantuml.com/plantuml/svg/{enc_uc}', headers={'User-Agent': 'Mozilla/5.0'})
svg_uc = urllib.request.urlopen(req_uc_svg).read()
with open('diagrams/usecase_diagram.svg', 'wb') as f:
    f.write(svg_uc)

req_uc_png = urllib.request.Request(f'https://www.plantuml.com/plantuml/png/{enc_uc}', headers={'User-Agent': 'Mozilla/5.0'})
png_uc = urllib.request.urlopen(req_uc_png).read()
with open('diagrams/usecase_diagram.png', 'wb') as f:
    f.write(png_uc)

print("Use Case Diagram generated successfully!")

print("=== 2. GENERATING SEQUENCE DIAGRAM (HIDE FOOTBOX) ===")
seq_puml = """@startuml
autonumber
hide footbox
skinparam sequence {
    ParticipantBackgroundColor #FF6B6B
    ParticipantBorderColor #C92A2A
    ParticipantFontColor white
    ParticipantFontSize 13
    LifeLineBorderColor #FA5252
    LifeLineBackgroundColor #8CE99A
    ArrowColor #2B2D42
    ArrowFontColor #10002B
    ActorBorderColor #2B2D42
    ActorBackgroundColor #EDF2F4
    GroupBorderColor #C92A2A
    GroupHeaderFontColor #C92A2A
}

actor Musician as "Musician"
participant UI as "Workspace UI"
participant Audio as "Web Audio Player"
participant Engine as "Client Harmonic Engine"
participant Server as "Backend System"
database DB as "Database"

Musician -> UI : Click Play button
activate UI
UI -> Audio : Start audio playback at current time
activate Audio
Audio --> UI : Return playback timestamp in milliseconds
deactivate Audio
UI -> UI : Highlight active bar on Smart Grid (< 100ms sync)
deactivate UI

Musician -> UI : Click chord cell to edit (Open Edit Marker)
activate UI
UI -> Musician : Display Edit Marker modal dialog
deactivate UI

Musician -> UI : Submit corrected chord (e.g. Root = C, Quality = maj9)
activate UI
UI -> Engine : Recalculate harmonic context (Chord = Cmaj9, Key = G)
activate Engine

alt Invalid Chord Form
    Engine --> UI : Return syntax validation error
    UI --> Musician : Display "Invalid Chord Format" warning message
else Valid Chord Form
    Engine --> UI : Return Roman degree (IVmaj9) and scale mode (C Lydian)
    deactivate Engine
    UI -> UI : Update Smart Grid display instantly (Zero Latency)

    Musician -> UI : Click "Save to Playlist"
    UI -> Server : Send update request with modified chord events
    activate Server
    Server -> DB : Update song chart timeline (PostgreSQL JSONB)
    activate DB

    alt Valid Save and Update
        DB --> Server : Return database confirmation (200 OK)
        Server --> UI : Confirm chart successfully persisted
        UI --> Musician : Display "Custom Chart Saved" toast notification
    else Failed Save
        DB --> Server : Return database error (500 Error)
        deactivate DB
        Server --> UI : Return internal server error response
        deactivate Server
        UI --> Musician : Display error alert and save chart to LocalStorage
    end
end
deactivate UI
@enduml"""

with open('diagrams/sequence_diagram.puml', 'w', encoding='utf-8') as f:
    f.write(seq_puml)

enc_seq = encode_plantuml(seq_puml)
req_seq_svg = urllib.request.Request(f'https://www.plantuml.com/plantuml/svg/{enc_seq}', headers={'User-Agent': 'Mozilla/5.0'})
svg_seq = urllib.request.urlopen(req_seq_svg).read()
with open('diagrams/sequence_diagram.svg', 'wb') as f:
    f.write(svg_seq)

req_seq_png = urllib.request.Request(f'https://www.plantuml.com/plantuml/png/{enc_seq}', headers={'User-Agent': 'Mozilla/5.0'})
png_seq = urllib.request.urlopen(req_seq_png).read()
with open('diagrams/sequence_diagram.png', 'wb') as f:
    f.write(png_seq)

print("Sequence Diagram generated successfully (footbox removed)!")

print("=== 3. GENERATING ACTIVITY DIAGRAM (TRUE DIAMOND & MERGE DIAMOND REMOVED) ===")
act_puml = """@startuml
skinparam conditionStyle insideDiamond
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
else (No)
endif

|Musician (User)|
:Practice with A/B loop, dynamic metronome, and speed shift;
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
@enduml"""

with open('diagrams/activity_diagram.puml', 'w', encoding='utf-8') as f:
    f.write(act_puml)

enc_act = encode_plantuml(act_puml)
req_act_svg = urllib.request.Request(f'https://www.plantuml.com/plantuml/svg/{enc_act}', headers={'User-Agent': 'Mozilla/5.0'})
svg_act = urllib.request.urlopen(req_act_svg).read().decode('utf-8')

# Patch SVG: Remove the merge diamond and connect both paths directly into top of Practice box
# 1. Target and remove merge diamond polygon
target_poly = '<polygon points="218.092,1027.175,230.092,1039.175,218.092,1051.175,206.092,1039.175,218.092,1027.175" fill="#E7F5FF" style="stroke:#1C7ED6;stroke-width:0.5;stroke-linejoin:miter;stroke-miterlimit:10;"/>'
svg_act = svg_act.replace(target_poly, '')

# 2. Extend the Yes branch straight down to y=1071.175 (top of Practice box)
old_top_arrow = '<polygon points="214.092,1017.175,218.092,1027.175,222.092,1017.175,218.092,1021.175" fill="#2B2D42" style="stroke:#2B2D42;stroke-width:1;stroke-linejoin:miter;stroke-miterlimit:10;"/>'
svg_act = svg_act.replace(old_top_arrow, '')
svg_act = svg_act.replace('<line x1="218.092" y1="1015.175" x2="218.092" y2="1027.175"', '<line x1="218.092" y1="1015.175" x2="218.092" y2="1071.175"')
svg_act = svg_act.replace('<line x1="218.092" y1="1051.175" x2="218.092" y2="1071.175" style="stroke:#2B2D42;stroke-width:1;"/>', '')

# 3. Connect No branch straight down into top of Practice box
old_no_line1 = '<line x1="39.572" y1="754.308" x2="39.572" y2="1039.175" style="stroke:#2B2D42;stroke-width:1;"/>'
new_no_line1 = '<line x1="39.572" y1="754.308" x2="39.572" y2="1071.175" style="stroke:#2B2D42;stroke-width:1;"/>'
svg_act = svg_act.replace(old_no_line1, new_no_line1)

old_no_line2 = '<line x1="39.572" y1="1039.175" x2="206.092" y2="1039.175" style="stroke:#2B2D42;stroke-width:1;"/>'
svg_act = svg_act.replace(old_no_line2, '')

old_no_arrow = '<polygon points="196.092,1035.175,206.092,1039.175,196.092,1043.175,200.092,1039.175" fill="#2B2D42" style="stroke:#2B2D42;stroke-width:1;stroke-linejoin:miter;stroke-miterlimit:10;"/>'
new_no_arrow = '<polygon points="35.572,1061.175,39.572,1071.175,43.572,1061.175,39.572,1065.175" fill="#2B2D42" style="stroke:#2B2D42;stroke-width:1;stroke-linejoin:miter;stroke-miterlimit:10;"/>'
svg_act = svg_act.replace(old_no_arrow, new_no_arrow)

with open('diagrams/activity_diagram.svg', 'w', encoding='utf-8') as f:
    f.write(svg_act)

# Render activity_diagram.png from patched SVG using resvg-js
try:
    cmd = 'npx -y @resvg/resvg-js-cli "diagrams/activity_diagram.svg" "diagrams/activity_diagram.png"'
    subprocess.run(cmd, shell=True, check=True)
    print("Activity Diagram SVG & PNG generated successfully with resvg!")
except Exception as e:
    print("Falling back to server png for activity diagram:", e)
    req_act_png = urllib.request.Request(f'https://www.plantuml.com/plantuml/png/{enc_act}', headers={'User-Agent': 'Mozilla/5.0'})
    png_act = urllib.request.urlopen(req_act_png).read()
    with open('diagrams/activity_diagram.png', 'wb') as f:
        f.write(png_act)

print("=== 4. UPDATING HTML VIEWER ===")
html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ChordSense Pro - Standard UML Diagrams</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; margin: 0; }
        h1 { text-align: center; color: #38bdf8; margin-bottom: 2rem; }
        .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 2.5rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); }
        .card h2 { color: #a78bfa; margin-top: 0; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }
        .img-container { text-align: center; background: #ffffff; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; overflow-x: auto; }
        img { max-width: 100%; height: auto; }
        .actions { margin-top: 1rem; display: flex; gap: 1rem; }
        .btn { padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; font-weight: 500; font-size: 0.875rem; background: #3b82f6; color: white; }
        .btn:hover { background: #2563eb; }
    </style>
</head>
<body>
    <h1>ChordSense Pro - Standard UML Specification Diagrams</h1>
    
    <div class="card">
        <h2>1. Use Case Diagram (10 Use Cases, 6 Actor Types)</h2>
        <div class="actions">
            <a class="btn" href="usecase_diagram.svg" download>Download SVG</a>
            <a class="btn" href="usecase_diagram.png" download>Download PNG</a>
        </div>
        <div class="img-container">
            <img src="usecase_diagram.svg" alt="Use Case Diagram" />
        </div>
    </div>

    <div class="card">
        <h2>2. Sequence Diagram (Clean Spacing, No Bottom Duplicate Footbox, Alt Frames)</h2>
        <div class="actions">
            <a class="btn" href="sequence_diagram.svg" download>Download SVG</a>
            <a class="btn" href="sequence_diagram.png" download>Download PNG</a>
        </div>
        <div class="img-container">
            <img src="sequence_diagram.svg" alt="Sequence Diagram" />
        </div>
    </div>

    <div class="card">
        <h2>3. Activity Diagram (True Diamond Decisions, Merge Diamond Removed, Swimlanes)</h2>
        <div class="actions">
            <a class="btn" href="activity_diagram.svg" download>Download SVG</a>
            <a class="btn" href="activity_diagram.png" download>Download PNG</a>
        </div>
        <div class="img-container">
            <img src="activity_diagram.svg" alt="Activity Diagram" />
        </div>
    </div>
</body>
</html>'''

with open('diagrams/index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("All diagrams regenerated and index.html updated successfully!")
