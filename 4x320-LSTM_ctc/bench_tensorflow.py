import tensorflow as tf
from support import toy_batch_ctc, default_params, write_results, print_results, plot_results, target_converter, sparse_tuple_from
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
import numpy as np

# Experiment_type
framework = 'tensorflow'
experiment = '4x320LSTM_CTC'

# Get data
bX, b_lenX, maskX, bY, b_lenY, classes = toy_batch_ctc()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, epochs = default_params()

# Create symbolic vars
x = tf.placeholder(tf.float32, [None, None, inp_dims])
x_len = tf.placeholder(tf.int32, [None])
y = tf.sparse_placeholder(tf.int32, [None])

weights = {'out': tf.Variable(tf.truncated_normal(shape=[2 * rnn_size, classes], stddev=0.1), name='W_out')}
biases = {'out': tf.Variable(tf.zeros([classes]), name='b_out')}


# Create network
def get_EESEN(x, rnn_size, weights, biases, x_len, classes):
    shape = tf.shape(x)
    batch_size, max_timesteps = shape[0], shape[1]

    with tf.name_scope('MultiLSTM'):
        fw_cell = [tf.nn.rnn_cell.LSTMCell(rnn_size) for _ in range(4)]
        bw_cell = [tf.nn.rnn_cell.LSTMCell(rnn_size) for _ in range(4)]

        h1, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=fw_cell, cells_bw=bw_cell,
                                                                  inputs=x, sequence_length=x_len, dtype=tf.float32)

    with tf.name_scope('Affine'):
        h1_rs = tf.reshape(h1, [-1, 2 * rnn_size])
        logits = tf.matmul(h1_rs, weights['out']) + biases['out']
        logits = tf.reshape(logits, [batch_size, max_timesteps, classes])
        logits = tf.transpose(logits, (1, 0, 2))

    return logits


pred = get_EESEN(x=x, rnn_size=rnn_size, weights=weights, biases=biases, x_len=x_len, classes=classes)


# Create loss, optimizer and train function
loss = tf.reduce_mean(tf.nn.ctc_loss(inputs=pred, labels=y, sequence_length=x_len, time_major=True))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_step = optimizer.minimize(loss)

# Initialize session
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = False

# Print parameter count
params = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    params += variable_parametes
print('# network parameters: ' + str(params))

# Start training
with tf.Session(config=config) as sess:
    sess.run(init)
    time = []
    # Convert labels
    bY = target_converter(bY, b_lenY)
    bY = sparse_tuple_from(bY)

    for i in range(epochs):
        print('Epoch {}/{}'.format(i, epochs))

        start = timer()
        _, output = sess.run([train_step, pred], feed_dict={x: bX, y: bY, x_len: b_lenX})
        end = timer()
        time.append(end - start)
        assert (output.shape == (seq_len, batch_size, classes))

write_results(script_name=os.path.basename(__file__), framework=framework, experiment=experiment, parameters=params, run_time=time)
print_results(time)

# Plot results
fig, ax =plot_results(time)
fig.savefig('{}_{}.pdf'.format(framework,experiment), bbox_inches='tight')