import lasagne
from data import toy_batch_ctc, default_params, write_results, print_results, plot_results
from timeit import default_timer as timer
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
import os
import ctc

# Get data
bX, b_lenX, maskX, bY, b_lenY,classes = toy_batch_ctc()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, epochs = default_params()

# Create symbolic vars
input_var = T.ftensor3('bX')
input_var_lens = T.ivector('b_lenX')
mask_var = T.matrix('maskX')
target_var = T.ivector('bY')
target_var_lens = T.ivector('b_lenY')


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

    h5 = non_flattening_dense(h4, batch_size=batch_size, seq_len=seq_len, num_units=classes,
                              nonlinearity=lasagne.nonlinearities.linear)
    return h5

def non_flattening_dense(l_in, batch_size, seq_len, *args, **kwargs):
    # Flatten down the dimensions for everything but the features
    l_flat = lasagne.layers.ReshapeLayer(l_in, (-1, [2]))
    # Make a dense layer connected to it
    l_dense = lasagne.layers.DenseLayer(l_flat, *args, **kwargs)
    # Reshape it back out
    l_nonflat = lasagne.layers.ReshapeLayer(l_dense, (batch_size, seq_len, l_dense.output_shape[1]))
    return l_nonflat

# Create network
network = get_bench_net_lstm(input_var=input_var,mask_var=mask_var, inp_dim=inp_dims, rnn_size=rnn_size, classes=classes)

# Create loss, optimizer and train function
prediction = lasagne.layers.get_output(network)
loss = T.mean(ctc.cpu_ctc_th(prediction, input_var_lens, target_var, target_var_lens))

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

fn_inputs = [input_var, input_var_lens, mask_var, target_var, target_var_lens]
train_fn = theano.function(fn_inputs, loss, updates=updates)

# Print parameter count
print('# network parameters: ' + str(lasagne.layers.count_params(network)))

# Start training
time=[]
for i in range(epochs):
    print('Epoch {}/{}'.format(i,epochs))
    start =timer()
    train_loss =  train_fn(bX, b_lenX, maskX, bY, b_lenY)
    end = timer()
    time.append(end-start)
write_results(os.path.basename(__file__), time)
print_results(time)

# Plot results
plot_results(time)
plt.show()