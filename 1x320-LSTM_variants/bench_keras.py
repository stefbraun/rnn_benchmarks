import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras.api.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.utils import to_categorical
from support import toy_batch, default_params, write_results, print_results, plot_results
from timeit import default_timer as timer
# import matplotlib.pyplot as plt
import os
import numpy as np
import editdistance

# Experiment_type
framework = 'keras'
experiment = '1x320LSTM'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, max_len, inp_dims = bX.shape
rnn_size, learning_rate, epochs = default_params()

# Create symbolic vars
x = Input(shape=(None, inp_dims), dtype='float32', name='input')

# Create network
fw_cell = LSTM(rnn_size, return_sequences=False)(x)
h3 = Dense(classes, activation='relu')(fw_cell)
model = Model(inputs=x, outputs=h3)
model.compile(optimizer='Adam',
              loss='categorical_crossentropy')
pred = model.predict_on_batch(x=bX)
print(pred.shape)


# Print parameter count
params = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    params += variable_parametes
print('# network parameters: ' + str(params))

time = []

start = timer()
model.fit(x=bX, y=to_categorical(bY, num_classes=classes),epochs=epochs)
end = timer()
time.append(end - start)
output = model.predict_on_batch(x=bX)
assert (output.shape == (batch_size, classes))
print(np.mean(time))

# time = []
# for i in range(epochs):
#     print('Epoch {}/{}'.format(i, epochs))
#     start = timer()
#     model.train_on_batch(x=bX, y=to_categorical(bY, num_classes=classes))
#     end = timer()
#     time.append(end - start)
#     output = model.predict_on_batch(x=bX)
#     assert (output.shape == (batch_size, classes))

print_results(time)

# # Plot results
# fig, ax = plot_results(time)
# fig.savefig('{}_{}.pdf'.format(framework, experiment), bbox_inches='tight')
