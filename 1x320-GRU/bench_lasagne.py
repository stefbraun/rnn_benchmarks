import lasagne
from data import toy_batch, default_params, write_results, print_results, plot_results
from timeit import default_timer as timer
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
import os

# Experiment_type
framework = 'lasagne'
experiment = '1x320GRU'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, epochs = default_params()

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
    h1 = lasagne.layers.GRULayer(l_in, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform())
    h2 = lasagne.layers.SliceLayer(h1, -1, axis=1)
    h3 = lasagne.layers.DenseLayer(h2, num_units=classes, nonlinearity=lasagne.nonlinearities.rectify)
    return h3


# Create network
network = get_bench_net(x, inp_dims, rnn_size)

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
write_results(script_name=os.path.basename(__file__), framework=framework, experiment=experiment, parameters=params, run_time=time)
print_results(time)
