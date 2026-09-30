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

test_uc = '''@startuml
left to right direction
skinparam packageStyle rectangle
skinparam actor {
    BorderColor #2B2D42
    BackgroundColor #EDF2F4
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

actor "Musician / Learner" as Musician
actor "Music Content Manager\\n(Business Admin)" as BizAdmin
actor "System & AI Ops Admin\\n(IT Admin)" as ITAdmin

actor "AI Pipeline Worker\\n(Demucs / MERT)" as AIWorker
actor "PostgreSQL Database" as DB

rectangle "ChordSense Pro System" {
    together {
        usecase "UC01: Ingest Audio & Detect Chords" as UC1
        usecase "Process AI Background Job" as UC1_Job
        usecase "UC02: Practice with Smart Grid & Loop" as UC2
        usecase "UC03: Inspect Roman Numerals & Scales" as UC3
        usecase "UC05: Audit & Edit Marker (HITL)" as UC5
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
@enduml'''

enc = encode_plantuml(test_uc)
req = urllib.request.Request(f'https://www.plantuml.com/plantuml/png/{enc}', headers={'User-Agent': 'Mozilla/5.0'})
data = urllib.request.urlopen(req).read()
with open('diagrams/test_uc_box.png', 'wb') as f:
    f.write(data)
print('Done test_uc_box with [norank]')
