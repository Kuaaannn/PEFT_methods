"""Read the manuscript's one color source; keep table labels semantic."""
from pathlib import Path
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]
PALETTE_PATH = ROOT / 'paper/palette.tex'
PALETTE_INPUTS = ('analysis/paper_palette.py', 'paper/palette.tex')
TABLE_STYLE_INPUTS = ('analysis/paper_palette.py',)


def read_colors(path=PALETTE_PATH):
    source = Path(path).read_text()
    definitions = re.findall(r'\\definecolor\{([A-Za-z]+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}', source)
    if not definitions or len(definitions) != len(set(name for name, _ in definitions)):
        raise ValueError('Palette must contain unique six-digit HTML color definitions')
    return {name: '#' + value.upper() for name, value in definitions}


PALETTE = read_colors()
METHOD_COLORS = {name: PALETTE[key] for name, key in
                 [('lora', 'LoRAColor'), ('oft', 'OFTColor'),
                  ('dora', 'DoRAColor'), ('pissa', 'PiSSAColor'),
                  ('milora', 'MiLoRAColor'), ('hra', 'HRAColor'),
                  ('full', 'FullFTColor')]}


def metadata():
    return {'palette_path': 'paper/palette.tex',
            'palette_sha256': hashlib.sha256(PALETTE_PATH.read_bytes()).hexdigest(),
            'colors': PALETTE}


def colorize_table_labels(source):
    """Color method names inside tables, leaving numeric keys intact.

    TeX macros resolve the current palette when the manuscript is compiled;
    table data therefore need not be recalculated just to change a color.
    """
    lines, in_table = [], False
    for line in source.splitlines(keepends=True):
        if re.search(r'\\begin\{(?:tabular\*?|longtable)\}', line):
            in_table = True
        if in_table:
            line = re.sub(r'(?<![A-Za-z_\\])(?:LoRA|OFT)(?![A-Za-z_])',
                          lambda m: r'\loraname{}' if m[0] == 'LoRA' else r'\oftname{}', line)
        lines.append(line)
        if re.search(r'\\end\{(?:tabular\*?|longtable)\}', line):
            in_table = False
    return ''.join(lines)
