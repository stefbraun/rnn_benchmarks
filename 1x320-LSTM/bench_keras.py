import os
import time as timer

import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.utils import to_categorical

from support import toy_batch, default_params, write_results, print_results, plot_results

# Experiment_type
bench = 'keras-{}_LSTM'.format(keras.backend.backend())
version=keras.__version__
experiment = '1x320-LSTM_cross-entropy'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, max_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# Create symbolic vars
x = Input(shape=(None, inp_dims), dtype='float32', name='input')

# Create network
fw_cell = LSTM(rnn_size, return_sequences=False)(x)
h3 = Dense(classes, activation=None)(fw_cell)
model = Model(inputs=x, outputs=h3)
model.compile(optimizer='Adam', loss='categorical_crossentropy')

# Print parameter count
params = model.count_params()
print('# network parameters: ' + str(params))

# Start training
time = []
for i in range(batches):
    print('Batch {}/{}'.format(i, batches))
    start = timer.perf_counter()
    model.train_on_batch(x=bX, y=to_categorical(bY, num_classes=classes))
    end = timer.perf_counter()
    time.append(end - start)
    output = model.predict_on_batch(x=bX)
    assert (output.shape == (batch_size, classes))

# Write results
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=time, version=version)
print_results(time)
