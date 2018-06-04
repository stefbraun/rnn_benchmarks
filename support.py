import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd


def default_params():
    rnn_size = 320
    learning_rate = 1e-3
    batches = 500
    return rnn_size, learning_rate, batches


def toy_batch(seed=11, shape=(64, 100, 123), classes=10):
    batch_size, max_len, features = shape
    np.random.seed(seed)

    # Samples
    bX = np.float32(np.random.uniform(-1, 1, (shape)))
    b_lenX = np.int32(np.ones(batch_size) * max_len)

    # Targets
    bY = np.int32(np.random.randint(low=0, high=classes - 1, size=batch_size))

    return bX, b_lenX, bY, classes


def toy_batch_ctc(seed=11, shape=(32, 1000, 123), classes=59):
    batch_size, max_len, features = shape
    np.random.seed(seed)

    # Samples
    bX = np.float32(np.random.uniform(-1, 1, (shape)))
    b_lenX = np.int32(np.linspace(max_len / 2, max_len, batch_size))
    # print(b_lenX)
    maskX = np.zeros((batch_size, max_len), dtype='float32')
    for i, len_sample in enumerate(b_lenX):
        maskX[i, :len_sample] = np.ones((1, len_sample))

    # Targets
    bY = np.int32(np.random.randint(low=1, high=classes - 1,
                                    size=batch_size * 100))  # remember warp-ctc: 0 is the blank label, tensorflow-ctc: -1 is the blank label
    b_lenY = np.int32(np.ones(batch_size) * 100)  # labels per sample comes from WSJ-si84

    return bX, b_lenX, maskX, bY, b_lenY, classes


def check_results(batch_loss_list, batch_time_list, train_start, train_end):

    # Initialize
    abort = 0

    # 0. Check if loss is numeric (not NAN and not inf)
    check_loss=[np.isfinite(loss) for loss in batch_loss_list]
    if False not in check_loss:
        print('>>> Loss check 1/2 passed: loss is finite {}'.format(np.unique(check_loss)))
    else:
        print('!!! Loss check 1/2 failed: loss is NOT finite {}'.format(np.unique(check_loss)))
        abort = 1

    # 1. Check if loss is decreasing
    check_loss=np.diff(batch_loss_list)
    if np.sum(check_loss)<0:
        print('>>> Loss check 2/2 passed: loss is globally decreasing')
    else:
        print('!!! Loss check 2/2 failed: loss is NOT globally decreasing')
        abort=1

    # 2. Check deviation between the full loop time and the sum of individual batches
    loop_time = train_end - train_start
    batch_time_sum = np.sum(batch_time_list)
    factor = loop_time / batch_time_sum
    deviation = np.abs((1 - factor) * 100)

    if deviation < 1:  # Less than 1% deviation
        print('>>> Timing check passed -  < 1% deviation between loop time and sum of batches ::: Loop time {:.3f} ::: Sum of batch times {:.3f} ::: Deviation [%] {:.3f}'.format(loop_time,
                                                                                                      batch_time_sum,
                                                                                                      deviation))
    else:
        print('!!! Timing check failed - Deviation > 1% ::: Loop time {:.3f} ::: Sum of batch times {:.3f} :::'
              ' Deviation [%] {:.3f}'.format(loop_time, batch_time_sum, deviation))
        abort=1

    if abort==1:
        sys.exit('!!! Abort benchmark.')
    print('=' * 100)


def write_results(script_name, bench, experiment, parameters, run_time, version=None,
                  logfile=None):

    if logfile == None:
        # Get path
        repo_path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(repo_path, 'results', 'conf')) as f:
            mode = f.readline().strip()

        logfile = os.path.join(repo_path, 'results', mode, 'results.csv')

    # Prepare header
    if os.path.isfile(logfile) == False:
        df = pd.DataFrame(index=None, columns=['name', 'bench', 'version', 'experiment', 'parameters', 'runtime'])
        df.to_csv(logfile, index=None)

    # Prepare new results
    row_list = []
    for rt in run_time:
        row = OrderedDict()
        row['experiment'] = experiment
        row['bench'] = bench
        row['version'] = version
        row['name'] = script_name
        row['parameters'] = parameters
        row['runtime'] = rt

        row_list.append(row)

    dfa = pd.DataFrame.from_dict(row_list)

    # Append new results
    df = pd.read_csv(logfile)
    df = df.append(dfa)
    df.to_csv(logfile, index=None)


def print_results(run_time):
    if len(run_time) > 100:
        run_time = run_time[100:]
    else:
        print('!!! First 100 batches are considered as warm-up. Please run more batches')
    run_time=np.asarray(run_time)*1000
    print(
        '>>> Time per batch [ms] ::: Mean {:.1f} ::: Std {:.1f} ::: Median {:.1f} ::: 99Percentile {:.1f} ::: Min {:.1f} ::: Max {:.1f}'.format(
            np.mean(run_time), np.std(run_time),
            np.median(run_time), np.percentile(run_time, 99), np.min(run_time), np.max(run_time)))

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
