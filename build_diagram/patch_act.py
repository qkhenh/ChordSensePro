with open('diagrams/test_act_raw.svg', 'r', encoding='utf-8') as f:
    svg = f.read()

# 1. Remove the 12x12 merge diamond polygon
target_poly = '<polygon points="218.092,1027.175,230.092,1039.175,218.092,1051.175,206.092,1039.175,218.092,1027.175" fill="#E7F5FF" style="stroke:#1C7ED6;stroke-width:0.5;stroke-linejoin:miter;stroke-miterlimit:10;"/>'
svg = svg.replace(target_poly, '')

# 2. Extend the Yes branch straight down to y=1071.175 (top of Practice box)
old_top_arrow = '<polygon points="214.092,1017.175,218.092,1027.175,222.092,1017.175,218.092,1021.175" fill="#2B2D42" style="stroke:#2B2D42;stroke-width:1;stroke-linejoin:miter;stroke-miterlimit:10;"/>'
svg = svg.replace(old_top_arrow, '')
svg = svg.replace('<line x1="218.092" y1="1015.175" x2="218.092" y2="1027.175"', '<line x1="218.092" y1="1015.175" x2="218.092" y2="1071.175"')
svg = svg.replace('<line x1="218.092" y1="1051.175" x2="218.092" y2="1071.175" style="stroke:#2B2D42;stroke-width:1;"/>', '')

# 3. For No branch:
# Old:
# <line x1="39.572" y1="754.308" x2="39.572" y2="1039.175" style="stroke:#2B2D42;stroke-width:1;"/>
# <line x1="39.572" y1="1039.175" x2="206.092" y2="1039.175" style="stroke:#2B2D42;stroke-width:1;"/>
# <polygon points="196.092,1035.175,206.092,1039.175,196.092,1043.175,200.092,1039.175" fill="#2B2D42" style="stroke:#2B2D42;stroke-width:1;stroke-linejoin:miter;stroke-miterlimit:10;"/>

# New: Line goes straight from (39.572, 754.308) down to (39.572, 1071.175) with arrowhead pointing down at y=1071.175!
old_no_line1 = '<line x1="39.572" y1="754.308" x2="39.572" y2="1039.175" style="stroke:#2B2D42;stroke-width:1;"/>'
new_no_line1 = '<line x1="39.572" y1="754.308" x2="39.572" y2="1071.175" style="stroke:#2B2D42;stroke-width:1;"/>'
svg = svg.replace(old_no_line1, new_no_line1)

old_no_line2 = '<line x1="39.572" y1="1039.175" x2="206.092" y2="1039.175" style="stroke:#2B2D42;stroke-width:1;"/>'
svg = svg.replace(old_no_line2, '')

old_no_arrow = '<polygon points="196.092,1035.175,206.092,1039.175,196.092,1043.175,200.092,1039.175" fill="#2B2D42" style="stroke:#2B2D42;stroke-width:1;stroke-linejoin:miter;stroke-miterlimit:10;"/>'
new_no_arrow = '<polygon points="35.572,1061.175,39.572,1071.175,43.572,1061.175,39.572,1065.175" fill="#2B2D42" style="stroke:#2B2D42;stroke-width:1;stroke-linejoin:miter;stroke-miterlimit:10;"/>'
svg = svg.replace(old_no_arrow, new_no_arrow)

with open('diagrams/test_act_patched.svg', 'w', encoding='utf-8') as f:
    f.write(svg)
print('Done test_act_patched.svg')
