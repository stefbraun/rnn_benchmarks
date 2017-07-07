import csv
import os.path

import matplotlib.pyplot as plt
import numpy as np


def toy_batch(seed=11, shape=(25, 1000, 123), classes=25):
    batch_size, max_len, features = shape
    np.random.seed(seed)

    # Samples
    bX = np.float32(np.random.uniform(-1, 1, (shape)))
    b_lenX = np.int32(np.ones(batch_size) * max_len)

    # Targets
    bY = np.int32(np.random.randint(low=0, high=classes - 1, size=batch_size))

    return bX, b_lenX, bY, classes


def toy_batch_ctc(seed=11, shape=(25, 1000, 123), classes=59):
    batch_size, max_len, features = shape
    np.random.seed(seed)

    # Samples
    bX = np.float32(np.random.uniform(-1, 1, (shape)))
    b_lenX = np.int32(np.linspace(max_len / 2, max_len, batch_size))
    print(b_lenX)
    maskX = np.zeros((batch_size, max_len), dtype='float32')
    for i, len_sample in enumerate(b_lenX):
        maskX[i, :len_sample] = np.ones((1, len_sample))

    # Targets
    bY = np.int32(np.random.randint(low=1, high=classes - 1,
                                    size=batch_size * 100))  # remember warp-ctc: 0 is the blank label, tensorflow-ctc: -1 is the blank label
    b_lenY = np.int32(np.ones(batch_size) * 100)  # labels per sample comes from WSJ-si84

    return bX, b_lenX, maskX, bY, b_lenY, classes


def default_params():
    rnn_size = 320
    learning_rate = 1e-3
    epochs = 500
    return rnn_size, learning_rate, epochs


def write_results(script_name, framework, experiment, parameters, run_time):
    logfile = '/media/stefbraun/ext4/Dropbox/repos/rnn_benchmarks/results/results.csv'
    # Write first line if logfile doesn't exits
    if os.path.isfile(logfile) == False:
        with open(logfile, 'a') as f:
            c = csv.writer(f)
            c.writerow(
                ['Name', 'Framework', 'Experiment', 'Parameters', 'Min', 'Max', 'Mean', 'Std', 'Median'])

    # Write data
    with open(logfile, 'a') as f:
        c = csv.writer(f)
        c.writerow([script_name, framework, experiment, parameters, np.min(run_time),
                    np.max(run_time), np.mean(run_time),
                    np.std(run_time), np.median(run_time)])


def print_results(run_time):
    print(
        'Min: {:.3f} Max: {:.3f} Mean: {:.3f} Median: {:.3f}'.format(np.min(run_time), np.max(run_time),
                                                                     np.mean(run_time),
                                                                     np.median(run_time)))


def plot_results(time):
    fig, ax = plt.subplots()
    ax.scatter(range(len(time)), time)
    ax.grid()
    ax.set_xlabel('Epoch #')
    ax.set_ylabel('Time per epoch [sec]')
    return fig, ax


def bar_chart(logfile='results/results_980ti.csv', category='Median', selection=[1, 2, 3], title='Time per epoch'):
    cat_dict = dict()
    cat = 0
    with open(logfile, 'rt') as f:
        f = csv.reader(f)
        experiments = []
        for idx, row in enumerate(f):
            if idx == 0:
                cats = row
            elif idx in selection:
                experiments.append(row)
    if len(selection) > 5:
        fig_width = 8 + 1.3 * len(selection) - 5
    else:
        fig_width = 8
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ind = np.arange(len(selection))
    width = 0.3
    x_labels = []
    y_bar = []
    for row in experiments:
        # X-axis
        cat_idx = cats.index('Framework')
        cat_name = row[cat_idx]
        x_labels.append(cat_name)

        # Y-axis
        y_idx = cats.index('Median')
        y_val = row[y_idx]
        y_bar.append(y_val)

    color_list = []
    processed_x_lables = []
    for label in x_labels:
        if 'tensorflow' in label:
            color_list.append('red')
        elif 'pytorch' in label:
            color_list.append('green')
        elif 'lasagne' in label:
            color_list.append('blue')
        else:
            color_list.append('deepskyblue')
        processed_x_lables.append('\n'.join(label.split('_')))

    color = ['red', 'green', 'blue']
    ax.bar(ind, y_bar, width=width, color=color_list)
    plt.grid()
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel('{} time per epoch [sec]'.format(category), fontsize=14)
    ax.set_xticks(ind)

    ax.set_xticklabels(processed_x_lables, rotation=0, fontsize=14)

    ax.set_title(title, fontsize=14)

    return fig, ax


# Helper functions for label conversion from warp-ctc to tf-ctc format:-(
def target_converter(bY, b_lenY):
    b_lenY_cs = np.cumsum(b_lenY)[:-1]
    bY_conv = np.split(bY, b_lenY_cs)
    return bY_conv


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape
