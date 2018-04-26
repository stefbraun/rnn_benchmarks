import os
import time as timer

import tensorflow as tf

from support import toy_batch, default_params, write_results, print_results, plot_results

# Experiment_type
bench = 'tensorflow_LSTMBlockCell'
version=tf.__version__
experiment = '1x320-LSTM_cross-entropy'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, max_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# Create symbolic vars
x = tf.placeholder(tf.float32, [None, None, inp_dims])
seq_len = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.int32, [None])

# Create network
fw_cell = tf.contrib.rnn.LSTMBlockCell(rnn_size)

h1, _ = tf.nn.dynamic_rnn(cell=fw_cell, inputs=x, sequence_length=seq_len, dtype=tf.float32)
h2 = h1[:, -1, :]
h3 = tf.layers.dense(h2, units=classes, activation=None)

# Create loss, optimizer and train function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h3, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_step = optimizer.minimize(loss)

# Initialize session
init = tf.global_variables_initializer()
config = tf.ConfigProto()
# config.gpu_options.allow_growth = False  # dynamic allocation of VRAM

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
    for i in range(batches):
        print('Batch {}/{}'.format(i, batches))
        start = timer.perf_counter()
        _, output = sess.run([train_step, h3], feed_dict={x: bX, y: bY, seq_len: b_lenX})
        end = timer.perf_counter()
        time.append(end - start)
        assert (output.shape == (batch_size, classes))

# Write results
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=time, version=version)
print_results(time)
