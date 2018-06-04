import os
import time as timer

import lasagne
import theano
import theano.tensor as T

from support import toy_batch, default_params, write_results, print_results, check_results

# Experiment_type
bench = 'lasagne_LSTMLayer'
version = lasagne.__version__
experiment = '4x320-BIDIR-LSTM_cross-entropy'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# Create symbolic vars
x = T.ftensor3('x')
y = T.ivector('y')


# Create network
def get_bench_net_lstm(input_var, inp_dim, rnn_size):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)

    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    h1f = lasagne.layers.LSTMLayer(l_in, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform())
    h1b = lasagne.layers.LSTMLayer(l_in, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform(), backwards=True)
    h1 = lasagne.layers.ConcatLayer([h1f, h1b], axis=2)

    h2f = lasagne.layers.LSTMLayer(h1, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform())
    h2b = lasagne.layers.LSTMLayer(h1, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform(), backwards=True)
    h2 = lasagne.layers.ConcatLayer([h2f, h2b], axis=2)

    h3f = lasagne.layers.LSTMLayer(h2, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform())
    h3b = lasagne.layers.LSTMLayer(h2, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform(), backwards=True)
    h3 = lasagne.layers.ConcatLayer([h3f, h3b], axis=2)

    h4f = lasagne.layers.LSTMLayer(h3, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform())
    h4b = lasagne.layers.LSTMLayer(h3, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform(), backwards=True)
    h4 = lasagne.layers.ConcatLayer([h4f, h4b], axis=2)

    h5 = lasagne.layers.SliceLayer(h4, -1, axis=1)
    h6 = lasagne.layers.DenseLayer(h5, num_units=classes, nonlinearity=lasagne.nonlinearities.softmax, b=None)

    return h6


# Create network
network = get_bench_net_lstm(x, inp_dims, rnn_size)

# Print parameter count
params = lasagne.layers.count_params(network)
print('>>> # network parameters: ' + str(params))

# Create loss, optimizer and train function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(predictions=prediction, targets=y)
loss = loss.mean()

update_params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.adam(loss, update_params, learning_rate=learning_rate)

fn_inputs = [x, y]

start = timer.perf_counter()
train_fn = theano.function(fn_inputs, loss, updates=updates)
prediction_det = lasagne.layers.get_output(network, deterministic=True)
output_fn = theano.function([x], prediction_det)
end = timer.perf_counter()
print('>>> Theano function compilation took {:.1f} seconds'.format(end - start))

# Check for correct sizes
assert (network.input_shape == (None, 2*rnn_size))  # final projection input size (Batch_size x rnn_size)
assert (network.W.eval().shape == (2*rnn_size, classes))  # final projection kernel size (rnn_size x classes)
output = output_fn(bX)
output_fn.sync_shared()
assert (output.shape == (batch_size, classes))  # output size

# Start training
batch_time = []
batch_loss = []
train_start = timer.perf_counter()  # start of training
for i in range(batches):
    batch_start = timer.perf_counter()  # start of batch
    loss = train_fn(bX, bY)
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
