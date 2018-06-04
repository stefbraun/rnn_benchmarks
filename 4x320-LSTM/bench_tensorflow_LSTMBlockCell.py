import os
import time as timer

import tensorflow as tf

from support import toy_batch, default_params, write_results, print_results, check_results

# Experiment_type
bench = 'tensorflow_LSTMBlockCell'
version = tf.__version__
experiment = '4x320-BIDIR-LSTM_cross-entropy'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, max_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# Create symbolic vars
x = tf.placeholder(tf.float32, [None, None, inp_dims])
x_len = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.int32, [None])

# Create network
fw_cell = [tf.contrib.rnn.LSTMBlockCell(rnn_size) for _ in range(4)]
bw_cell = [tf.contrib.rnn.LSTMBlockCell(rnn_size) for _ in range(4)]

h1, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=fw_cell, cells_bw=bw_cell,
                                                          inputs=x, sequence_length=x_len, dtype=tf.float32)
h2 = h1[:, -1, :]
h3 = tf.layers.dense(h2, units=classes, activation=None, use_bias=False)

# Create loss, optimizer and train function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h3, labels=y))
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
    # Check for correct sizes
    assert (h2._shape_as_list() == [None, 2*rnn_size])  # final projection input size (rnn_size)
    assert (tf.trainable_variables(scope='dense/kernel:0')[0].shape.as_list() == [2*rnn_size, classes])  # final projection output size (rnn_size, classes)
    output = sess.run(h3, feed_dict={x: bX, y: bY, x_len: b_lenX})
    assert (output.shape == (batch_size, classes))

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
