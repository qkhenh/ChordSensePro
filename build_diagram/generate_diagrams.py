import os
import zlib
import urllib.request

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

# 1. USE CASE DIAGRAM (Exact match to Figure 2: Vertical Box in Center, Actors on Left & Right)
usecase_puml = """@startuml
left to right direction
skinparam packageStyle rectangle
skinparam actor {
    BorderColor #2B2D42
    BackgroundColor #EDF2F4
    FontColor #10002B
    FontSize 13
}
skinparam usecase {
    BackgroundColor #D0BFFF
    BorderColor #5A189A
    FontColor #10002B
    FontSize 12
}
skinparam rectangle {
    BackgroundColor #F8F0FC
    BorderColor #9775FA
    FontColor #3C096C
}
skinparam arrow {
    Color #2B2D42
    FontColor #495057
}

actor "Musician / Learner" as Musician
actor "Music Content Manager\\n(Business Admin)" as BizAdmin
actor "System & AI Ops Admin\\n(IT Admin)" as ITAdmin

actor "AI Pipeline Worker\\n(Demucs / MERT)" as AIWorker
actor "PostgreSQL Database" as DB

rectangle "ChordSense Pro System" {
    together {
        usecase "UC01: Ingest Audio & Detect Chords" as UC1
        usecase "UC03: Inspect Roman Numerals & Scales" as UC3
        usecase "UC05: Audit & Edit Marker (HITL)" as UC5
        usecase "UC02: Practice with Smart Grid & Loop" as UC2
        usecase "Process AI Background Job" as UC1_Job
        usecase "UC04: Manage Custom Charts & Playlists" as UC4
        usecase "UC07: Verify Public Song Charts" as UC7
        usecase "UC06: Manage Chord Dictionary" as UC6
        usecase "UC08: Monitor GPU Queues & Workers" as UC8
        usecase "UC09: Manage Users, Roles & Limits" as UC9
    }

    UC1 .[norank].> UC1_Job : <<include>>
    UC2 .[norank].> UC3 : <<include>>
    UC7 .[norank].> UC5 : <<include>>
    UC5 .[norank].> UC2 : <<extend>>
}

Musician -- UC1
Musician -- UC2
Musician -- UC4

BizAdmin -- UC6
BizAdmin -- UC7

ITAdmin -- UC8
ITAdmin -- UC9

UC1_Job -- AIWorker
UC4 -- DB
UC6 -- DB
UC9 -- DB
@enduml"""

# 2. SEQUENCE DIAGRAM (Clean spacing, clear terms, no squished text, generic bar cell selection)
seq_puml = """@startuml
autonumber
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

# 3. ACTIVITY DIAGRAM (Exact match to Figure 1: Swimlanes / Partitions format + English)
act_puml = """@startuml
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

diagrams = [
    ('usecase_diagram', usecase_puml),
    ('sequence_diagram', seq_puml),
    ('activity_diagram', act_puml)
]

for name, puml in diagrams:
    enc = encode_plantuml(puml)
    with open(f'diagrams/{name}.puml', 'w', encoding='utf-8') as f:
        f.write(puml)
    
    svg_url = f'https://www.plantuml.com/plantuml/svg/{enc}'
    req = urllib.request.Request(svg_url, headers={'User-Agent': 'Mozilla/5.0'})
    svg_data = urllib.request.urlopen(req).read()
    with open(f'diagrams/{name}.svg', 'wb') as f:
        f.write(svg_data)
    
    png_url = f'https://www.plantuml.com/plantuml/png/{enc}'
    req_png = urllib.request.Request(png_url, headers={'User-Agent': 'Mozilla/5.0'})
    png_data = urllib.request.urlopen(req_png).read()
    with open(f'diagrams/{name}.png', 'wb') as f:
        f.write(png_data)
    
    print(f'Successfully generated {name}.svg and {name}.png')

# HTML viewer
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
        <h2>1. Use Case Diagram (Center System Box, Actors on Left & Right, Strict UML)</h2>
        <div class="actions">
            <a class="btn" href="usecase_diagram.svg" download>Download SVG</a>
            <a class="btn" href="usecase_diagram.png" download>Download PNG</a>
        </div>
        <div class="img-container">
            <img src="usecase_diagram.svg" alt="Use Case Diagram" />
        </div>
    </div>

    <div class="card">
        <h2>2. Sequence Diagram (Clear Spacing, Domain Terms, Alt Frames)</h2>
        <div class="actions">
            <a class="btn" href="sequence_diagram.svg" download>Download SVG</a>
            <a class="btn" href="sequence_diagram.png" download>Download PNG</a>
        </div>
        <div class="img-container">
            <img src="sequence_diagram.svg" alt="Sequence Diagram" />
        </div>
    </div>

    <div class="card">
        <h2>3. Activity Diagram (Swimlanes / Partitions Format in English)</h2>
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

print('Updated diagrams/index.html successfully!')
