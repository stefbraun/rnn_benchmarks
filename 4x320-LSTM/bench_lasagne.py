import os
from timeit import default_timer as timer

import lasagne
import theano
import theano.tensor as T

from support import toy_batch, default_params, write_results, print_results, plot_results

# Experiment_type
framework = 'lasagne'
experiment = '4x320LSTM'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, epochs = default_params()

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
    h6 = lasagne.layers.DenseLayer(h5, num_units=classes, nonlinearity=lasagne.nonlinearities.rectify)

    return h6


# Create network
network = get_bench_net_lstm(x, inp_dims, rnn_size)

# Create loss, optimizer and train function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(predictions=prediction, targets=y)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

fn_inputs = [x, y]
train_fn = theano.function(fn_inputs, loss, updates=updates)
network_output = lasagne.layers.get_output(network, deterministic=True)
output_fn = theano.function([x], network_output)

# Print parameter count
params = lasagne.layers.count_params(network)
print('# network parameters: ' + str(params))

# Start training
time = []
for i in range(epochs):
    print('Epoch {}/{}'.format(i, epochs))
    start = timer()
    train_loss = train_fn(bX, bY)
    end = timer()
    time.append(end - start)
    output = output_fn(bX)
    assert (output.shape == (batch_size, classes))
write_results(script_name=os.path.basename(__file__), framework=framework, experiment=experiment, parameters=params,
              run_time=time)
print_results(time)

# Plot results
fig, ax = plot_results(time)
fig.savefig('{}_{}.pdf'.format(framework, experiment), bbox_inches='tight')
