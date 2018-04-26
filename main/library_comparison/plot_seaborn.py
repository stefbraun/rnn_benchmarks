import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
from pathlib import Path

sns.set_style('darkgrid')

def linebreak(row):
    row['bench']='\n'.join(row['bench'].split('_'))
    return row

def get_color_palette(unique_benchs):
    colors=[]
    for bench in unique_benchs:
        if 'tensorflow' in bench:
            c="#3498db"
        if 'pytorch' in bench:
            c="#e74c3c"
        if 'lasagne' in bench:
            c="#95a5a6"
        if 'keras' in bench:
            c="#2ecc71"
        colors.append(c)
    return colors

# Load file
repo_path = Path(__file__).resolve().parents[2]
logfile = os.path.join(repo_path, 'results', 'library_comparison', 'results.csv')
df = pd.read_csv(logfile)

# Parameters
experiments =  list(df['experiment'].unique())

# Plots
# fig, axs = plt.subplots(2, 2, figsize=(16, 9))

# for exp, ax in zip(experiments, axs.reshape(-1)):
for exp in experiments:

    dfp = df[df['experiment'] == exp]
    dfp=dfp.apply(linebreak, axis=1)

    height = len(dfp['bench'].unique()) / 1.5
    print(height)
    fig, ax = plt.subplots(1, figsize=(8, height))

    colors = get_color_palette(dfp['bench'].unique())
    sns.set_palette(colors)

    sns.barplot(ax=ax,data=dfp, y='bench', x='runtime', capsize=0.02)
    ax.set_title(exp)
    ax.set_xlabel('Time per batch [sec]')



    min_width=200
    max_width=0
    for p, c in zip(ax.patches, colors):
        min_width=np.min([min_width, p.get_width()])
        max_width=np.max([max_width, p.get_width()])

    ax.set_xlim((0, 1.5*max_width))
    max_x = np.max(ax.get_xlim())

    for p, c in zip(ax.patches, colors):

        print(max_x)
        ax.text(p.get_width()+max_x/7, p.get_y()+p.get_height()/1.3, '{:5.2f}sec // {:5.2f}x'.format(p.get_width(), p.get_width()/min_width),
                fontsize=12, fontweight='bold', color=c, ha='center', va='bottom')

    ax.set_ylabel('')
    output_file= os.path.join(repo_path, 'results/library_comparison/{}'.format(exp))
    fig.savefig(output_file, bbox_inches='tight')

plt.show()
