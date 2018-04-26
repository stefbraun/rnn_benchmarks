import csv
import os.path

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from collections import OrderedDict

def toy_batch(seed=11, shape=(25, 1000, 123), classes=20):
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
    batches = 1000
    return rnn_size, learning_rate, batches


def write_results(script_name, bench, experiment, parameters, run_time, version=None):

    # Get path
    repo_path = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(repo_path, 'results', 'conf')) as f:
        mode = f.readline().strip()

    logfile = os.path.join(repo_path, 'results', mode, 'results.csv')

    # Prepare header
    if os.path.isfile(logfile) == False:
        df=pd.DataFrame(index=None, columns=['name', 'bench', 'version','experiment', 'parameters', 'runtime'])
        df.to_csv(logfile, index=None)

    # Prepare new results
    row_list=[]
    for rt in run_time:
        row=OrderedDict()
        row['experiment'] = experiment
        row['bench'] = bench
        row['version'] = version
        row['name'] = script_name
        row['parameters'] = parameters
        row['runtime'] = rt

        row_list.append(row)

    dfa= pd.DataFrame.from_dict(row_list)

    # Append new results
    df = pd.read_csv(logfile)
    df=df.append(dfa)
    df.to_csv(logfile, index=None)

def print_results(run_time):
    print(
        'Min: {:.3f} Max: {:.3f} Mean: {:.3f} Median: {:.3f}'.format(np.min(run_time), np.max(run_time),
                                                                     np.mean(run_time),
                                                                     np.median(run_time)))


def plot_results(time):
    fig, ax = plt.subplots()
    ax.scatter(range(len(time)), time)
    ax.grid()
    ax.set_xlabel('Batch #')
    ax.set_ylabel('Time per Batch [sec]')
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
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape
