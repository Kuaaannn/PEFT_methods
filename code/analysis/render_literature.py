from pathlib import Path
import csv, json, os, re
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import argparse
parser = argparse.ArgumentParser(description="Render the PEFT controls figure")
parser.add_argument("--paper", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
PAPER, SOURCE, OUT = args.paper, args.source, args.output
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads(SOURCE.read_text())['summary']
palette = dict(re.findall(r'\\definecolor\{(\w+)\}\{HTML\}\{([\da-fA-F]{6})\}', (PAPER / 'palette.tex').read_text()))
colors = ['#' + palette[x] for x in ('AccentColor', 'NLLReductionColor', 'BaseColor')]
categories = ['documented', 'explicitly_not_met', 'unclear_not_reported']

# The PGF PDF backend uses the manuscript's pdfLaTeX/T1 Computer Modern text.
plt.rcParams.update({
    'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
    'pgf.preamble': r'\usepackage[T1]{fontenc}\usepackage{fix-cm}',
    'font.family': 'serif', 'text.usetex': True, 'font.size': 9,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 9,
    'text.color': 'black', 'axes.axisbelow': True, 'axes.linewidth': .55,
})
fig = plt.figure(figsize=(6.5, 3.05), facecolor='white')
panels = [
    (['replication','lr_selection','lr_space','matched'],
     ['Training runs\nand uncertainty','LR selection','LR candidates','Comparable\nconditions'],
     [1,5,9,13], [.21, .20, .27, .72],
     .008, .97, r'\textbf{(a)} Experimental controls'),
    (['backbone','data','exposure','modules','budget'],
     ['Checkpoint','Data','Exposure','Adapted modules','Trainable budget'],
     [1,4,7,10,13], [.715, .20, .265, .72],
     .515, .97, r'\textbf{(b)} Comparison conditions'),
]
records = []
label_positions = []
for index, (keys, labels, positions, bounds, title_x, title_y, title) in enumerate(panels):
    ax = fig.add_axes(bounds)
    fig.text(title_x, title_y, title, fontsize=9.3, va='center')
    for y, key in zip(positions, keys):
        counts = summary[key]
        for offset, cat, color in zip([-.75,0,.75], categories, colors):
            n = counts[cat]
            width = n
            ax.barh(y+offset, width, height=.58, color=color,
                    edgecolor='none', zorder=3)
            ax.text(width+1.5, y+offset, r'\textbf{' + str(n) + '}',
                    ha='left', va='center', fontsize=8.0, color='black', zorder=4)
            label_positions.append({'panel':index+1, 'criterion':key,
                                    'category':cat, 'count':n, 'bar_length':width})
        records.append({'panel':index+1, 'criterion':key, **counts})
    ax.set_yticks(positions, [f"{label} ({summary[k]['n']})" for k, label in zip(keys, labels)])
    ax.set_ylim(14.35, -.35)
    ax.set_xlim(0,72)
    ax.set_xticks([0,32,64])
    ax.grid(axis='x', color='#'+palette['GuideColor'], linewidth=.45)
    ax.tick_params(axis='y', length=0, pad=5, labelsize=8.5)
    ax.tick_params(axis='x', length=2.5, width=.5, pad=3)
    for side in ('top','left','right'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#'+palette['SpineColor'])
fig.text(.59, .102, 'Number of papers', ha='center', va='center', fontsize=9)
fig.legend([Patch(facecolor=x) for x in colors],
           ['Documented','Explicit departure','Unclear'], loc='lower center',
           bbox_to_anchor=(.51,.008), ncol=3, frameon=False, fontsize=9,
           handlelength=1.25, handleheight=.75, columnspacing=1.5,
           handletextpad=.5, borderaxespad=0)
fig.savefig(OUT / 'peft_controls_review.pdf',
            metadata={'Title':'Experimental controls in PEFT comparisons','Author':'','CreationDate':None})
fig.savefig(OUT / 'peft_controls_review.pgf')
with (OUT / 'figure_data.csv').open('w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]))
    w.writeheader(); w.writerows(records)
(OUT / 'count_labels.json').write_text(json.dumps(label_positions, indent=2)+'\n')
print('Figure 2 rendered with three horizontal category bars per criterion and counts to the right')
