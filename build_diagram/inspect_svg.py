with open('diagrams/activity_diagram.svg', 'r', encoding='utf-8') as f:
    svg = f.read()

import re
polygons = re.findall(r'<polygon[^>]+/>', svg)
print('Total polygons:', len(polygons))
for p in polygons:
    print(p)
