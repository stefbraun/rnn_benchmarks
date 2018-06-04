import os
import time as timer

import lasagne
import theano
import theano.tensor as T
from theano.tensor.nnet.ctc import (ctc)

from support import toy_batch_ctc, default_params, write_results, print_results, check_results

# Experiment_type
bench = 'lasagne_LSTMLayer'
version = lasagne.__version__
experiment = '4x320-BIDIR-LSTM_CTC'

# Get data
bX, b_lenX, maskX, bY, b_lenY, classes = toy_batch_ctc()
bY = bY.reshape(-1, b_lenY.max())  # compatibility with theano ctc interface
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# Create symbolic vars
input_var = T.ftensor3('bX')
input_var_lens = T.ivector('b_lenX')
mask_var = T.matrix('maskX')
target_var = T.imatrix('bY')


# Create network
def get_bench_net_lstm(input_var, mask_var, inp_dim, rnn_size, classes):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)

    # Masking layer
    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var)

    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    h1f = lasagne.layers.LSTMLayer(l_in, num_units=rnn_size, mask_input=l_mask, hid_init=lasagne.init.GlorotUniform())
    h1b = lasagne.layers.LSTMLayer(l_in, num_units=rnn_size, mask_input=l_mask, hid_init=lasagne.init.GlorotUniform(),
                                   backwards=True)
    h1 = lasagne.layers.ConcatLayer([h1f, h1b], axis=2)

    h2f = lasagne.layers.LSTMLayer(h1, num_units=rnn_size, mask_input=l_mask, hid_init=lasagne.init.GlorotUniform())
    h2b = lasagne.layers.LSTMLayer(h1, num_units=rnn_size, mask_input=l_mask, hid_init=lasagne.init.GlorotUniform(),
                                   backwards=True)
    h2 = lasagne.layers.ConcatLayer([h2f, h2b], axis=2)

    h3f = lasagne.layers.LSTMLayer(h2, num_units=rnn_size, mask_input=l_mask, hid_init=lasagne.init.GlorotUniform())
    h3b = lasagne.layers.LSTMLayer(h2, num_units=rnn_size, mask_input=l_mask, hid_init=lasagne.init.GlorotUniform(),
                                   backwards=True)
    h3 = lasagne.layers.ConcatLayer([h3f, h3b], axis=2)

    h4f = lasagne.layers.LSTMLayer(h3, num_units=rnn_size, mask_input=l_mask, hid_init=lasagne.init.GlorotUniform())
    h4b = lasagne.layers.LSTMLayer(h3, num_units=rnn_size, mask_input=l_mask, hid_init=lasagne.init.GlorotUniform(),
                                   backwards=True)
    h4 = lasagne.layers.ConcatLayer([h4f, h4b], axis=2)

    h5 = non_flattening_dense(h4, batch_size=batch_size, seq_len=seq_len, num_units=classes,
                              nonlinearity=lasagne.nonlinearities.linear)

    h6 = lasagne.layers.DimshuffleLayer(h5, (1, 0, 2))

    return h6


def non_flattening_dense(l_in, batch_size, seq_len, *args, **kwargs):
    # Flatten down the dimensions for everything but the features
    l_flat = lasagne.layers.ReshapeLayer(l_in, (-1, [2]))
    # Make a dense layer connected to it
    l_dense = lasagne.layers.DenseLayer(l_flat, b=None,  *args, **kwargs,)
    # Reshape it back out
    l_nonflat = lasagne.layers.ReshapeLayer(l_dense, (batch_size, seq_len, l_dense.output_shape[1]))
    return l_nonflat


# Create network
network = get_bench_net_lstm(input_var=input_var, mask_var=mask_var, inp_dim=inp_dims, rnn_size=rnn_size,
                             classes=classes)

# Create loss, optimizer and train function
prediction = lasagne.layers.get_output(network)
loss = T.mean(ctc(prediction, target_var, input_var_lens))
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
fn_inputs = [input_var, input_var_lens, mask_var, target_var]

start = timer.perf_counter()
train_fn = theano.function(fn_inputs, loss, updates=updates)
output_fn = theano.function([input_var, mask_var], prediction)
end = timer.perf_counter()
print('>>> Theano function compilation took {:.1f} seconds'.format(end - start))

# Print parameter count
params = lasagne.layers.count_params(network)
print('# network parameters: ' + str(params))

# Check for correct sizes
output_layer = network.input_layer.input_layer
assert (output_layer.input_shape == (None, 2*rnn_size))  # final projection input size (Batch_size x rnn_size)
assert (output_layer.W.eval().shape == (2*rnn_size, classes))  # final projection kernel size (rnn_size x classes)
output = output_fn(bX, maskX)
output_fn.sync_shared()
assert (output.shape == (seq_len, batch_size, classes))  # output size

# Start training
batch_time = []
batch_loss = []
train_start = timer.perf_counter()  # start of training
for i in range(batches):
    batch_start = timer.perf_counter()  # start of batch
    loss = train_fn(bX, b_lenX, maskX, bY)
    train_fn.sync_shared() # synchronize function call for precise time measurement
    batch_end = timer.perf_counter()  # end of batch
    batch_time.append(batch_end - batch_start)
    batch_loss.append(loss)
train_end = timer.perf_counter() # end of training

# Results handling
print_results(batch_time)
check_results(batch_loss, batch_time, train_start, train_end)
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=batch_time, version=version)