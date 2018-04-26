import os
import time as timer

import lasagne
import theano
import theano.tensor as T

from support import toy_batch, default_params, write_results, print_results, plot_results

# Experiment_type
bench = 'lasagne_default-LSTM'
version = lasagne.__version__
experiment = '1x320-LSTM_cross-entropy'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# Create symbolic vars
x = T.ftensor3('x')
y = T.ivector('y')


# Create network
def get_bench_net(input_var, inp_dim, rnn_size):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)

    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    h1 = lasagne.layers.LSTMLayer(l_in, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform())
    h2 = lasagne.layers.SliceLayer(h1, -1, axis=1)
    h3 = lasagne.layers.DenseLayer(h2, num_units=classes, nonlinearity=lasagne.nonlinearities.linear)
    return h3


# Create network
network = get_bench_net(x, inp_dims, rnn_size)

# Print parameter count
params = lasagne.layers.count_params(network)
print('# network parameters: ' + str(params))

# Create loss, optimizer and train function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(predictions=prediction, targets=y)
loss = loss.mean()

update_params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.adam(loss, update_params, learning_rate=learning_rate)

fn_inputs = [x, y]
train_fn = theano.function(fn_inputs, loss, updates=updates)
network_output = lasagne.layers.get_output(network, deterministic=True)
output_fn = theano.function([x], network_output)

# Start training
time = []
for i in range(batches):
    print('Batch {}/{}'.format(i, batches))
    start = timer.perf_counter()
    train_loss = train_fn(bX, bY)
    end = timer.perf_counter()
    time.append(end - start)
    output = output_fn(bX)
    assert (output.shape == (batch_size, classes))

# Write results
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=time, version=version)
print_results(time)