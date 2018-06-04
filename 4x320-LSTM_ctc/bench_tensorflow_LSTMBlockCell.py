import os
import time as timer

import tensorflow as tf

from support import toy_batch_ctc, default_params, write_results, print_results, target_converter, \
    sparse_tuple_from, check_results

# Experiment_type
bench = 'tensorflow_LSTMBlockCell'
version = tf.__version__
experiment = '4x320-BIDIR-LSTM_CTC'

# Get data
bX, b_lenX, maskX, bY, b_lenY, classes = toy_batch_ctc()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# Create symbolic vars
x = tf.placeholder(tf.float32, [None, None, inp_dims])
x_len = tf.placeholder(tf.int32, [None])
y = tf.sparse_placeholder(tf.int32)

weights = {'out': tf.Variable(tf.truncated_normal(shape=[2 * rnn_size, classes], stddev=0.1), name='W_out')}

# Create network
def get_EESEN(x, rnn_size, weights, x_len, classes):
    shape = tf.shape(x)
    batch_size, max_timesteps = shape[0], shape[1]

    with tf.name_scope('MultiLSTM'):
        fw_cell = [tf.contrib.rnn.LSTMBlockCell(rnn_size) for _ in range(4)]
        bw_cell = [tf.contrib.rnn.LSTMBlockCell(rnn_size) for _ in range(4)]

        h1, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=fw_cell, cells_bw=bw_cell,
                                                                  inputs=x, sequence_length=x_len, dtype=tf.float32)

    with tf.name_scope('Affine'):
        h1_rs = tf.reshape(h1, [-1, 2 * rnn_size])
        logits = tf.matmul(h1_rs, weights['out'])
        logits = tf.reshape(logits, [batch_size, max_timesteps, classes])
        logits = tf.transpose(logits, (1, 0, 2))

    return logits, h1


pred, h1 = get_EESEN(x=x, rnn_size=rnn_size, weights=weights, x_len=x_len, classes=classes)

# Create loss, optimizer and train function
loss = tf.reduce_mean(tf.nn.ctc_loss(inputs=pred, labels=y, sequence_length=x_len, time_major=True))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_step = optimizer.minimize(loss)

# Initialize session
init = tf.global_variables_initializer()
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

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

with tf.Session(config=config) as sess:
    sess.run(init)
    bY = target_converter(bY, b_lenY)
    bY = sparse_tuple_from(bY)

    # Check for correct sizes
    assert (h1._shape_as_list() == [None, None, 2*rnn_size])  # final projection input size (rnn_size)
    assert (weights['out'].shape.as_list() == [2*rnn_size, classes])  # final projection kernel size (rnn_size, classes)
    output = sess.run(pred, feed_dict={x: bX, y: bY, x_len: b_lenX})
    assert (output.shape == (seq_len, batch_size, classes))

    # Start training
    batch_time = []
    batch_loss = []
    train_start=timer.perf_counter()
    for i in range(batches):
        batch_start = timer.perf_counter()
        _, loss_val = sess.run([train_step, loss], feed_dict={x: bX, y: bY, x_len: b_lenX})
        batch_end = timer.perf_counter()
        batch_time.append(batch_end - batch_start)
        batch_loss.append(loss_val)
    train_end = timer.perf_counter()

# Results handling
print_results(batch_time)
check_results(batch_loss, batch_time, train_start, train_end)
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=batch_time, version=version)
