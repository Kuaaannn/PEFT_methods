"""Shared colors for experiment plots."""
import hashlib
from pathlib import Path

PALETTE = {
    'LoRAColor': '#425B76',
    'OFTColor': '#B76E4A',
    'DoRAColor': '#3F7C73',
    'PiSSAColor': '#A85C68',
    'MiLoRAColor': '#9B8448',
    'HRAColor': '#78627F',
    'FullFTColor': '#748092',
    'RestoredColor': '#268978',
    'InkColor': '#000000',
    'GuideColor': '#DDE2E8',
    'SpineColor': '#414A57',
}
METHOD_COLORS = {name: PALETTE[key] for name, key in
                 [('lora', 'LoRAColor'), ('oft', 'OFTColor'),
                  ('dora', 'DoRAColor'), ('pissa', 'PiSSAColor'),
                  ('milora', 'MiLoRAColor'), ('hra', 'HRAColor'),
                  ('full', 'FullFTColor')]}
PALETTE_INPUTS = ('analysis/plot_palette.py',)


def metadata():
    return {'palette_path': PALETTE_INPUTS[0],
            'palette_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'colors': PALETTE}
