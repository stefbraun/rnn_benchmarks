import os
import time as timer

import tensorflow as tf

from support import toy_batch, default_params, write_results, print_results, plot_results
import numpy as np

# Experiment_type
bench = 'tensorflow_cudnnLSTM'
version=tf.__version__
experiment = '4x320-BIDIR-LSTM_cross-entropy'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, max_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# cudnn compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))

# Create symbolic vars
x = tf.placeholder(tf.float32, [None, None, inp_dims])
x_len = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.int32, [None])

# Create network
cudnn_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=4, num_units=rnn_size, direction='bidirectional')
h1, _ = cudnn_lstm(x)
h2 = h1[-1, :, :]
h3 = tf.layers.dense(h2, units=classes, activation=None)

# Create loss, optimizer and train function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h3, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_step = optimizer.minimize(loss)

# Initialize session
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Print parameter count
params = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    if 'cudnn_lstm' in str(variable):
        biases = cudnn_lstm.canonical_bias_shapes
        weights = cudnn_lstm.canonical_weight_shapes
        all_biases = np.sum(biases)
        all_weights = np.sum([t[0]*t[1] for t in weights])
        params += all_biases
        params += all_weights
    else:
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
        _, output = sess.run([train_step, h3], feed_dict={x: bX, y: bY, x_len: b_lenX})
        end = timer.perf_counter()
        time.append(end - start)
        assert (output.shape == (batch_size, classes))

# Write results
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=time, version=version)
print_results(time)