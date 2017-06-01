import lasagne
from data import toy_batch, default_params, write_results, print_results
from timeit import default_timer as timer
import numpy as np
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
import os

bX, bY, b_lenX, maskX = toy_batch()
inp_dims = bX.shape[2]
rnn_size, learning_rate, epochs = default_params()

# Create symbolic vars
x = T.ftensor3('x')
mask = T.matrix('mask')
y = T.ivector('y')

# Create network
def get_bench_net_lstm(input_var, mask_var, inp_dim, rnn_size):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)

    # Masking layer
    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var)

    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    h1f = lasagne.layers.LSTMLayer(l_in, num_units=rnn_size, mask_input=l_mask, hid_init=lasagne.init.GlorotUniform())
    h1b = lasagne.layers.LSTMLayer(l_in, num_units=rnn_size, mask_input=l_mask,hid_init=lasagne.init.GlorotUniform(),backwards=True)
    h1 = lasagne.layers.ConcatLayer([h1f, h1b], axis=2)

    h2f = lasagne.layers.LSTMLayer(h1, num_units=rnn_size, mask_input=l_mask,hid_init=lasagne.init.GlorotUniform())
    h2b = lasagne.layers.LSTMLayer(h1, num_units=rnn_size, mask_input=l_mask,hid_init=lasagne.init.GlorotUniform(), backwards = True)
    h2 = lasagne.layers.ConcatLayer([h2f, h2b], axis=2)

    h3f = lasagne.layers.LSTMLayer(h2, num_units=rnn_size, mask_input=l_mask,hid_init=lasagne.init.GlorotUniform())
    h3b = lasagne.layers.LSTMLayer(h2, num_units=rnn_size, mask_input=l_mask,hid_init=lasagne.init.GlorotUniform(), backwards = True)
    h3 = lasagne.layers.ConcatLayer([h3f, h3b], axis=2)

    h4f = lasagne.layers.LSTMLayer(h3, num_units=rnn_size, mask_input=l_mask,hid_init=lasagne.init.GlorotUniform())
    h4b = lasagne.layers.LSTMLayer(h3, num_units=rnn_size, mask_input=l_mask,hid_init=lasagne.init.GlorotUniform(), backwards = True)
    h4 = lasagne.layers.ConcatLayer([h4f, h4b], axis=2)

    h5 = lasagne.layers.SliceLayer(h4, -1, axis=1)
    return h5

# get network
print('Compiling network')
network = get_bench_net_lstm(x, mask, inp_dims, rnn_size)

# create loss, optimizer and train function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(predictions=prediction, targets=y)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

fn_inputs = [x, y, mask]
train_fn = theano.function(fn_inputs, loss, updates=updates)

print('# network parameters: ' + str(lasagne.layers.count_params(network)))

time=[]
for i in range(epochs):
    start =timer()
    train_loss =  train_fn(bX, bY, maskX)
    end = timer()
    time.append(end-start)
write_results(os.path.basename(__file__), time)
print_results(time)

plt.scatter(range(len(time)), time)
plt.grid()
plt.show()