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

uc_clean = '''@startuml
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
    FontSize 12
}
skinparam rectangle {
    BackgroundColor #F8F9FA
    BorderColor #ADB5BD
    FontColor #212529
}
skinparam arrow {
    Color #868E96
    FontColor #495057
}

' Left Actors
actor "Musician / Learner" as Musician
actor "Music Content Manager\\n(Business Admin)" as BizAdmin
actor "System & AI Ops Admin\\n(IT Admin)" as ITAdmin

' Right Actors
actor "AI Pipeline Worker\\n(Demucs / MERT)" as AIWorker
actor "PostgreSQL Database" as DB

rectangle "ChordSense Pro System" {
    package "Core User & Admin Features" {
        usecase "UC01: Ingest Audio Track" as UC1
        usecase "UC02: Practice Workspace (Smart Grid & Loop)" as UC2
        usecase "UC07: Verify Public Song Charts" as UC7
        usecase "UC04: Manage Charts & Playlists" as UC4
        usecase "UC06: Manage Chord Dictionary" as UC6
        usecase "UC09: Manage Users, Roles & Limits" as UC9
        usecase "UC08: Monitor GPU Queues & Workers" as UC8
    }

    package "AI & Harmonic Sub-services" {
        usecase "Process AI Background Job" as UC1_Job
        usecase "UC03: Inspect Roman Numerals & Scales" as UC3
        usecase "UC05: Audit & Edit Marker (HITL)" as UC5
        usecase "Persist Chart & User Data" as UC_DB
    }

    UC1 ..> UC1_Job : <<include>>
    UC2 ..> UC3 : <<include>>
    UC5 ..> UC2 : <<extend>>
    UC7 ..> UC5 : <<include>>

    UC4 ..> UC_DB : <<include>>
    UC6 ..> UC_DB : <<include>>
    UC9 ..> UC_DB : <<include>>
}

' Left Actor Connections
Musician --> UC1
Musician --> UC2
Musician --> UC4

BizAdmin --> UC7
BizAdmin --> UC6

ITAdmin --> UC9
ITAdmin --> UC8

' Right Actor Connections
UC1_Job --> AIWorker
UC_DB --> DB

@enduml'''

enc = encode_plantuml(uc_clean)
req = urllib.request.Request(f'https://www.plantuml.com/plantuml/png/{enc}', headers={'User-Agent': 'Mozilla/5.0'})
png = urllib.request.urlopen(req).read()
with open('diagrams/test_uc_clean2.png', 'wb') as f:
    f.write(png)
print('Done test_uc_clean2.png')
