import lasagne
from data import toy_batch, default_params, write_results, print_results, plot_results
from timeit import default_timer as timer
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
import os

# Get data
bX, bY, b_lenX, maskX = toy_batch()
inp_dims = bX.shape[2]
rnn_size, learning_rate, epochs = default_params()

# Create symbolic vars
x = T.ftensor3('x')
mask = T.matrix('mask')
y = T.ivector('y')


# Create network
def get_bench_net(input_var, mask_var, inp_dim, rnn_size):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)

    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    h1 = lasagne.layers.GRULayer(l_in, num_units=rnn_size, hid_init=lasagne.init.GlorotUniform())
    h2 = lasagne.layers.SliceLayer(h1, -1, axis=1)
    return h2


# Create network
network = get_bench_net(x, mask, inp_dims, rnn_size)

# Create loss, optimizer and train function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(predictions=prediction, targets=y)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

fn_inputs = [x, y]
train_fn = theano.function(fn_inputs, loss, updates=updates)

# Print parameter count
print('# network parameters: ' + str(lasagne.layers.count_params(network)))

# Start training
time = []
for i in range(epochs):
    start = timer()
    train_loss = train_fn(bX, bY)
    end = timer()
    time.append(end - start)
write_results(os.path.basename(__file__), time)
print_results(time)

# Plot results
plot_results(time)
plt.show()
