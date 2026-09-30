with open('diagrams/test_act_raw.svg', 'r', encoding='utf-8') as f:
    svg = f.read()

import re
lines = svg.splitlines()
for i, line in enumerate(lines):
    if '1027' in line or '1039' in line or '1051' in line or '1060' in line or '1070' in line:
        print(f'{i}: {line}')
