import numpy as np
import csv
import matplotlib.pyplot as plt

def toy_batch(seed=11, shape=(25,1000,123)):
    batch_size, max_len, features = shape
    np.random.seed(seed)
    bX = np.float32(np.random.uniform(-1, 1, (shape)))
    bY = np.int32(range(batch_size))
    b_lenX = np.linspace(max_len/2,max_len, batch_size, dtype=int)#np.arange(max_len/2, max_len/2+batch_size)
    maskX = np.zeros((batch_size, max_len), dtype='float32')
    for i, len_sample in enumerate(b_lenX):
        maskX[i, :len_sample] = np.ones((1, len_sample))

    return bX, np.asarray(bY), b_lenX, maskX

def default_params():
    rnn_size = 320
    learning_rate = 1e-3
    epochs = 50
    return rnn_size, learning_rate, epochs

def write_results(script_name, run_time):
    with open('results/results.csv','a') as f:
        c = csv.writer(f)
        c.writerow(['Name', 'Min[10:]','Min', 'Max', 'Mean', 'Std','Median'])
        c.writerow([script_name, np.min(np.sort(run_time)[10:]),np.min(run_time), np.max(run_time), np.mean(run_time),
                                                                       np.std(run_time),  np.median(run_time)])

def print_results(run_time):
    print('Min: {:.3f} Max: {:.3f} Mean: {:.3f} Median: {:.3f}'.format(np.min(run_time), np.max(run_time), np.mean(run_time),
                                                                       np.median(run_time)))

def plot_results(time):
    plt.scatter(range(len(time)), time)
    plt.grid()
    plt.xlabel('Epoch #')
    plt.ylabel('Time per epoch [sec]')

def bar_chart(logfile='results/results.csv',category='Median',selection=[0,1,2], title='Time per epoch'):
    cat_dict = dict()
    cat = 0
    with open(logfile, 'rb') as f:
        f = csv.reader(f)
        cat_rows = []
        for idx, row in enumerate(f):
            if (idx != 0) and row[0] == 'Name':
                cat_dict['{}'.format(cat)] = np.asarray(cat_rows)
                cat = cat + 1
                cat_rows = []
            cat_rows.append(row)
        cat_dict['{}'.format(cat)] = np.asarray(cat_rows)

    fig, ax = plt.subplots()
    ind = np.arange(len(selection))
    width=0.3
    x_labels=[]
    y_bar=[]
    for key in selection:
        key = str(key)
        cat_name = cat_dict[key][1][0].astype(str)
        x_labels.append(cat_name)

        cat_idx = np.where(cat_dict[key][0] == category)[0][0]
        cat_val = cat_dict[key][1][cat_idx]
        y_bar.append(cat_val)


    ax.bar(ind, y_bar, width=width, color='deepskyblue')
    plt.grid()
    ax.set_ylabel('{} time per epoch [sec]'.format(category), fontsize=18)
    ax.set_xticks(ind)
    ax.set_xticklabels(x_labels, rotation=0,fontsize=18)
    ax.set_title(title, fontsize=18)

    return fig, ax