import tensorflow as tf
from data import toy_batch, default_params, write_results, print_results
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

bX, bY, b_lenX, maskX = toy_batch()
inp_dims = bX.shape[2]
rnn_size, learning_rate, epochs = default_params()

# Create symbolic vars
x = tf.placeholder(tf.float32, [None, None, inp_dims])
seq_len = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.int32, [None])
seqlen = tf.placeholder(tf.int32, [None])

# Create network
fw_cell = tf.contrib.rnn.LSTMCell(rnn_size)
bw_cell = tf.contrib.rnn.LSTMCell(rnn_size)

final_hidden, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=[fw_cell]*4, cells_bw=[bw_cell]*4, inputs=x, sequence_length=seq_len, dtype=tf.float32)
hidden_last = last_relevant(final_hidden, seq_len)
#
# # create loss
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hidden_last, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# create optimizer
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# Print parameter count
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
print(total_parameters)

with tf.Session(config=config) as sess:
    sess.run(init)
    time=[]
    for i in tqdm(range(epochs)):
        start =timer()
        _, output = sess.run([train_step, hidden_last], feed_dict={x: bX, y: bY, seq_len: b_lenX})
        end = timer()
        time.append(end-start)

write_results(os.path.basename(__file__), time)
print_results(time)

print(output.shape)
plt.scatter(range(len(time)), time)
plt.grid()
plt.show()