import numpy as np
import csv
import matplotlib.pyplot as plt
import os.path


def toy_batch(seed=11, shape=(25, 1000, 123), classes=15):
    batch_size, max_len, features = shape
    np.random.seed(seed)

    # Samples
    bX = np.float32(np.random.uniform(-1, 1, (shape)))
    b_lenX = np.int32(np.ones(batch_size) * max_len)

    # Targets
    bY = np.int32(np.random.randint(low=0, high=classes, size=batch_size))

    return bX, b_lenX, bY, classes


def toy_batch_ctc(seed=11, shape=(25, 1000, 123), classes=58):
    batch_size, max_len, features = shape
    np.random.seed(seed)

    # Samples
    bX = np.float32(np.random.uniform(-1, 1, (shape)))
    b_lenX = np.int32(np.linspace(max_len / 2, max_len, batch_size))
    maskX = np.zeros((batch_size, max_len), dtype='float32')
    for i, len_sample in enumerate(b_lenX):
        maskX[i, :len_sample] = np.ones((1, len_sample))

    # Targets
    bY = np.int32(np.random.randint(low=1, high=classes + 1,
                                    size=batch_size * 100))  # remember warp-ctc: 0 is the blank label WTF.
    b_lenY = np.int32(np.ones(batch_size) * 100)  # labels per sample comes from WSJ-si84

    return bX, b_lenX, maskX, bY, b_lenY, classes


def default_params():
    rnn_size = 320
    learning_rate = 1e-3
    epochs = 50
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
    plt.scatter(range(len(time)), time)
    plt.grid()
    plt.xlabel('Epoch #')
    plt.ylabel('Time per epoch [sec]')


def bar_chart(logfile='results/results.csv', category='Median', selection=[1,2,3], title='Time per epoch'):
    cat_dict = dict()
    cat = 0
    with open(logfile, 'rb') as f:
        f = csv.reader(f)
        experiments = []
        for idx, row in enumerate(f):
            if idx == 0:
                cats = row
            elif idx in selection:
                experiments.append(row)


    fig, ax = plt.subplots()
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

    ax.bar(ind, y_bar, width=width, color='deepskyblue')
    plt.grid()
    ax.set_ylabel('{} time per epoch [sec]'.format(category), fontsize=16)
    ax.set_xticks(ind)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=16)
    ax.set_title(title, fontsize=16)

    return fig, ax
