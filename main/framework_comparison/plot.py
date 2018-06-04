import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sns.set_style('darkgrid')

import matplotlib.pylab as pylab

lparams = ['legend.fontsize', 'axes.labelsize', 'axes.titlesize', 'xtick.labelsize', 'ytick.labelsize']
fontsize = 11.5
params = {key: fontsize for key in lparams}
pylab.rcParams.update(params)

def match_case(row):
    for old, new in [('pytorch', 'PyTorch'), ('tensorflow', 'TensorFlow'), ('lasagne', 'Lasagne'),
                     ('keras', 'Keras'), ('theano', 'Theano'), ('cudnnLSTM', 'cuDNNLSTM')]:
        row['bench'] = row['bench'].replace(old, new)
    return row


def linebreak(row):
    row['bench'] = '\n'.join(row['bench'].split('_'))
    return row


def framework(row):
    row['framework'] = row['bench'].split('_')[0]
    if 'keras' in row['bench']:
        row['framework'] = 'keras'
    elif 'Keras' in row['bench']:
        row['framework'] = 'Keras'
    return row


def get_color_palette(unique_benchs):
    colors = []
    for bench in unique_benchs:
        if ('tensorflow' in bench) or ('TensorFlow' in bench):
            c = "#377eb8"
            c = '#4c72b0'
        if ('pytorch' in bench) or ('PyTorch' in bench):
            c = "#e41a1c"
            # c='#C44E52'
            c='#de2d26'
        if ('lasagne' in bench) or ('Lasagne' in bench):
            c = "#696969"
        if ('keras' in bench) or ('Keras' in bench):
            c = "#4daf4a"
            c= '#55A868'
        colors.append(c)
    return colors


# Load file
repo_path = Path(__file__).resolve().parents[2]
logfile = os.path.join(repo_path, 'results', 'framework_comparison', 'results.csv')
df = pd.read_csv(logfile)

# Parameters
experiments = list(df['experiment'].unique())

# for exp, ax in zip(experiments, axs.reshape(-1)):
for exp in experiments:

    dfp = df[df['experiment'] == exp]
    dfp = dfp.apply(match_case, axis=1)
    dfp = dfp.apply(framework, axis=1)
    dfp = dfp.apply(linebreak, axis=1)
    dfp = dfp.groupby('bench').tail(400)
    dfp['mean'] = dfp.groupby('bench').transform('mean')['runtime']
    dfp = dfp.sort_values(['mean'], ascending=True)
    dfp['runtime'] = dfp['runtime'] * 1000

    # Uber-plotting skillz: ax control
    fig_width = 8
    ax_height = len(dfp['bench'].unique()) * 0.5

    left_inch = 1.75
    left_rel = left_inch / fig_width
    ax_width_rel = 1 - left_rel - 0.005

    bottom_inch = 0.45
    top_inch = 0.2
    fig_height = ax_height + bottom_inch + top_inch

    bottom_rel = bottom_inch / fig_height
    ax_height_rel = (fig_height - bottom_inch - top_inch - 0.01) / fig_height

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_pad = (left_rel, bottom_rel, ax_width_rel, ax_height_rel)

    ax = fig.add_axes((ax_pad))

    # Start plotting
    colors = get_color_palette(dfp['bench'].unique())
    sns.set_palette(colors)

    custom_lines = [Line2D([0], [0], color=c, lw=4) for c in list(pd.unique(colors))]
    ax.legend(custom_lines, list(dfp['framework'].unique()))

    # dfp = dfp.apply(unbreak, axis=1)
    sns.barplot(ax=ax, data=dfp, y='bench', x='runtime', ci='sd')
    ax.set_title(exp.replace('_', '-'))
    ax.set_xlabel('Time per batch [milliseconds]')

    min_width = 1e6
    max_width = 0
    for p, c in zip(ax.patches, colors):
        min_width = np.min([min_width, p.get_width()])
        max_width = np.max([max_width, p.get_width()])

    ax.set_xlim((0, 1.6 * max_width))
    max_x = np.max(ax.get_xlim())

    for p, c in zip(ax.patches, colors):
        # print(max_x)
        if min_width > 10:
            ax.text(p.get_width() + max_x / 8, p.get_y() + p.get_height() / 1.3,
                    '{:4.0f}ms ::: {:3.1f}x'.format(p.get_width(), p.get_width() / min_width),
                    fontsize=fontsize+1.5, fontweight='bold', color=c, ha='center', va='bottom')
        else:
            ax.text(p.get_width() + max_x / 8, p.get_y() + p.get_height() / 1.3,
                    '{:4.1f}ms ::: {:3.1f}x'.format(p.get_width(), p.get_width() / min_width),
                    fontsize=fontsize+1.5, fontweight='bold', color=c, ha='center', va='bottom')

    ax.set_ylabel('')
    ax.xaxis.set_major_locator(MaxNLocator(prune='upper'))
    plt.setp(ax.get_xticklabels()[-1], visible=False)

    output_file = os.path.join(repo_path, 'results/framework_comparison/{}'.format(exp))

    fig.savefig(output_file, dpi=300)
    fig.savefig(output_file + '.pdf', dpi=300)
