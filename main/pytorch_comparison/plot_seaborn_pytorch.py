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

def get_color_palette():
    colors = list(reversed(['#fcbba1','#fc9272','#fb6a4a','#de2d26']))
    return colors

# Load file
repo_path = Path(__file__).resolve().parents[2]
logfile = os.path.join(repo_path, 'results', 'pytorch_comparison', 'results.csv')
df = pd.read_csv(logfile)

# Parameters
experiments =  list(df['experiment'].unique())

# for exp, ax in zip(experiments, axs.reshape(-1)):
for exp in experiments:

    dfp = df[df['experiment'] == exp]
    dfp = dfp.sort_values('version', ascending=False)
    dfp=dfp.apply(linebreak, axis=1)

    height = len(df['version'].unique())*len(dfp['bench'].unique()) / 1.5
    print(height)
    fig, ax = plt.subplots(1, figsize=(8, height))

    colors = get_color_palette()
    sns.set_palette(colors)

    sns.barplot(ax=ax,data=dfp, y='bench', x='runtime', hue='version', capsize=0.02)
    ax.set_title(exp)
    ax.set_xlabel('Time per batch [sec]')


    all_width=[p.get_width() for p in ax.patches]
    min_width=np.min(all_width)
    max_width=np.max(all_width)

    ax.set_xlim((0, 2.0*max_width))
    max_x = np.max(ax.get_xlim())

    for p in ax.patches:

        print(max_x)
        ax.text(p.get_width()+max_x/7, p.get_y()+p.get_height()/1.3, '{:5.2f}sec // {:5.2f}x'.format(p.get_width(), p.get_width()/min_width),
                fontsize=12, fontweight='bold', color='grey', ha='center', va='bottom')

    ax.set_ylabel('')

    # ax.legend(loc=2, bbox_to_anchor=(1.1, 1.05))
    ax.legend(loc=1)

    output_file= os.path.join(repo_path, 'results/pytorch_comparison/{}'.format(exp))
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')

plt.show()
