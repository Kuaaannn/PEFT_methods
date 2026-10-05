"""Shared publication typography and visual hierarchy; no empirical calculations."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib as mpl
import numpy as np
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.ticker import LogLocator
from plot_palette import PALETTE, METHOD_COLORS, PALETTE_INPUTS, metadata as palette_metadata

COLORS = METHOD_COLORS
RESTORED = PALETTE['RestoredColor']
INK = PALETTE['InkColor']
GUIDE = PALETTE['GuideColor']
SPINE = PALETTE['SpineColor']
STYLE_INPUTS = ('analysis/figure_style.py', *PALETTE_INPUTS)


def configure():
    # Some machines have both a minimal TinyTeX and a complete TeX installation.
    # Select a working installed toolchain without changing the user's setup.
    candidates = [Path(shutil.which('latex') or '/missing').parent,
                  Path('/Library/TeX/texbin')]
    selected = None
    for folder in dict.fromkeys(candidates):
        if not all((folder / name).exists() for name in ['latex', 'kpsewhich', 'dvipng']):
            continue
        if all(subprocess.run([str(folder/'kpsewhich'), package], capture_output=True).stdout.strip()
               for package in ['times.sty', 'type1cm.sty', 'underscore.sty']):
            selected = folder
            break
    if selected is None:
        raise RuntimeError('Figure typography requires latex, dvipng, times, type1cm and underscore; install a complete TeX distribution.')
    os.environ['PATH'] = str(selected) + os.pathsep + os.environ.get('PATH', '')
    mpl.rcParams.update({
        'text.usetex': True,
        # Use Times text with Computer Modern mathematics.
        'font.family': 'serif', 'font.serif': ['serif'],
        'font.sans-serif': ['Helvetica'], 'font.monospace': ['Courier'],
        'text.latex.preamble': r'\usepackage{times}',
        'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 8,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': SPINE,
        'axes.linewidth': .55, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'axes.labelpad': 4,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.major.width': .5, 'ytick.major.width': .5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.minor.width': .35, 'ytick.minor.width': .35,
        'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'lines.linewidth': 1.2, 'lines.markersize': 3.6,
        'lines.solid_capstyle': 'round', 'lines.dash_capstyle': 'butt',
        'legend.frameon': False, 'legend.borderpad': .2,
        'legend.handlelength': 1.6, 'legend.handletextpad': .5,
        'legend.columnspacing': 1.5, 'legend.labelspacing': .4,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'axes.unicode_minus': False,
        'grid.color': GUIDE, 'grid.linewidth': .45, 'grid.linestyle': '-',
        'grid.alpha': 1,
    })
    _install_data_audit()


def data_signature(fig):
    """Fingerprint plotted coordinates and ranges, independent of appearance."""
    digest = hashlib.sha256()
    def add(value):
        array = np.asarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    for ax in fig.axes:
        digest.update((ax.get_xscale()+'|'+ax.get_yscale()).encode())
        add(ax.get_xlim()); add(ax.get_ylim())
        for line in ax.lines: add(line.get_xydata())
        for collection in ax.collections:
            add(collection.get_offsets())
            for path in collection.get_paths(): add(path.vertices)
            if hasattr(collection, 'get_segments'):
                for segment in collection.get_segments(): add(segment)
            if collection.get_array() is not None: add(collection.get_array())
        for patch in ax.patches:
            add(patch.get_path().vertices)
            add(patch.get_patch_transform().get_matrix())
        for image in ax.images: add(image.get_array()); add(image.get_extent())
    return digest.hexdigest()


def _install_data_audit():
    """Optional save-time guard used for presentation-only revisions."""
    from matplotlib.figure import Figure
    if not os.environ.get('LORA_OFT_STYLE_AUDIT') or getattr(Figure.savefig, '_style_audit', False):
        return
    original = Figure.savefig
    def savefig(fig, filename, *args, **kwargs):
        before = getattr(fig, '_publication_data_signature', data_signature(fig))
        assert before == data_signature(fig), 'Plot coordinates changed after styling'
        result = original(fig, filename, *args, **kwargs)
        assert before == data_signature(fig), 'Plot coordinates changed while saving'
        audit = Path(os.environ['LORA_OFT_STYLE_AUDIT'])
        audit.parent.mkdir(parents=True, exist_ok=True)
        with audit.open('a') as handle:
            handle.write(json.dumps({'file':str(filename), 'axes':len(fig.axes),
                                    'coordinate_sha256':before, 'data_unchanged':True})+'\n')
        return result
    savefig._style_audit = True
    Figure.savefig = savefig


def _legend_colors(legend):
    if legend is None: return
    for text in legend.get_texts():
        label = re.sub(r'\\[A-Za-z]+|[{}$]', '', text.get_text()).lower()
        for method in ['milora', 'pissa', 'dora', 'lora', 'oft', 'hra']:
            if re.search(r'(?<![a-z])'+method+r'(?![a-z])', label):
                text.set_color(COLORS[method]); break


def black_errorbars(fig):
    """Use opaque black uncertainty stems and caps, preserving mean colors."""
    for ax in fig.axes:
        for container in ax.containers:
            if isinstance(container, ErrorbarContainer):
                _, caps, ranges = container.lines
                for line in caps:
                    line.set_color('black')
                    line.set_markeredgecolor('black')
                    line.set_alpha(1)
                for collection in ranges:
                    collection.set_color('black')
                    collection.set_alpha(1)
    legends = [*fig.legends, *[ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]]
    for legend in legends:
        for collection in legend.findobj(LineCollection):
            collection.set_color('black')
            collection.set_alpha(1)
        for line in legend.findobj(mpl.lines.Line2D):
            if line.get_marker() in ['_', '|']:
                line.set_color('black')
                line.set_markeredgecolor('black')
                line.set_alpha(1)


def prepare(fig):
    """Apply common finishing after figure-specific layout and plotting."""
    signature = data_signature(fig) if os.environ.get('LORA_OFT_STYLE_AUDIT') else None
    black_errorbars(fig)
    for ax in fig.axes:
        if ax.get_label() == 'response-matrix':
            continue
        ax.grid(False, which='both')
        # Continuous scatter plots benefit from both guides. For trajectories
        # and bar charts, guide the measured quantity without a cage of lines.
        if ax.axison and ax.get_label() != '<colorbar>' and not ax.images:
            bars = [c for c in ax.containers if isinstance(c, BarContainer)]
            scatter = any(isinstance(c, PathCollection) for c in ax.collections)
            direction = 'x' if bars and bars[0].orientation == 'horizontal' else ('both' if scatter else 'y')
            ax.set_axisbelow(True)
            ax.grid(axis=direction, which='major', color=GUIDE, linewidth=.45, linestyle='-', alpha=1)
        for spine in ax.spines.values():
            spine.set_color(SPINE)
            spine.set_linewidth(.55)
        ax.tick_params(which='major', width=.5, length=2.5, pad=3)
        ax.tick_params(which='minor', width=.35, length=1.5)
        if not ax.spines['left'].get_visible():
            ax.tick_params(axis='y',which='both',length=0)
        for axis, scale in [(ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())]:
            if scale == 'log':
                axis.set_minor_locator(LogLocator(base=10, subs=[2, 5]))
        for title in [ax.title, ax._left_title, ax._right_title]:
            match = re.match(r'^\(([a-z])\)\s*(.*)', title.get_text())
            if match:
                title.set_text(r'\textbf{('+match[1]+r')}\quad '+match[2])
            title.set_fontsize(8.5)
        # Keep uncertainty endpoints and caps, with less visual weight than
        # the means. Do not change marker sizes, offsets, bounds or visibility.
        uncertainty = set()
        for container in ax.containers:
            if isinstance(container, ErrorbarContainer):
                central, caps, ranges = container.lines
                for line in caps:
                    uncertainty.add(id(line)); line.set_markeredgewidth(.6)
                for collection in ranges:
                    collection.set_linewidth(.6)
        for line in ax.lines:
            if id(line) not in uncertainty and line.get_linestyle() not in ['None', 'none', '', ' ']:
                if .7 <= line.get_linewidth() <= 1.2: line.set_linewidth(line.get_linewidth()*1.12)
    for text in fig.findobj(mpl.text.Text):
        text.set_color(INK)
        # In normal Matplotlib labels '%' was literal; LaTeX requires escaping.
        text.set_text(re.sub(r'(?<!\\)%', r'\\%', text.get_text()))
        text.set_usetex(True)
    for legend in fig.legends: _legend_colors(legend)
    for ax in fig.axes: _legend_colors(ax.get_legend())
    if signature is not None:
        assert signature == data_signature(fig), 'Styling altered plotted coordinates or axis ranges'
        fig._publication_data_signature = signature


def metadata():
    return {**palette_metadata(), 'theme': 'Scientific palette with Times and Computer Modern',
            'text_engine': 'LaTeX', 'preamble': r'\usepackage{times}',
            'method_colors': COLORS, 'theme_path': STYLE_INPUTS[0],
            'theme_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
