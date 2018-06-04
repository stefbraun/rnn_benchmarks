import os
import time as timer

import lasagne
import theano
import theano.tensor as T

from support import toy_batch, default_params, write_results, print_results, check_results

# Experiment_type
bench = 'lasagne_LSTMLayer'
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
network = lasagne.layers.InputLayer(shape=(None, None, inp_dims), input_var=x)  # Input layer
network = lasagne.layers.LSTMLayer(network, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform())  # RNN layer
network = lasagne.layers.SliceLayer(network, -1, axis=1)  # slice last time step
network = lasagne.layers.DenseLayer(network, num_units=classes, nonlinearity=lasagne.nonlinearities.softmax,
                                    b=None)  # Output projection

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
prediction_det = lasagne.layers.get_output(network, deterministic=True)

start = timer.perf_counter()
train_fn = theano.function(fn_inputs, loss, updates=updates)
output_fn = theano.function([x], prediction_det)
end = timer.perf_counter()
print('>>> Theano function compilation took {:.1f} seconds'.format(end - start))

# Check for correct sizes
assert (network.input_shape == (None, rnn_size))  # final projection input size (Batch_size x rnn_size)
assert (network.W.eval().shape == (rnn_size, classes))  # final projection kernel size (rnn_size x classes)
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
