import os
import time as timer

import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.utils import to_categorical

from support import toy_batch, default_params, write_results, print_results, check_results

# Experiment_type
bench = 'keras-{}_LSTM'.format(keras.backend.backend())
version = keras.__version__
experiment = '1x320-LSTM_cross-entropy'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, max_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# Create symbolic vars
x = Input(shape=(None, inp_dims), dtype='float32', name='input')

# Create network
fw_cell = LSTM(rnn_size, return_sequences=False, implementation=2)(x)
h3 = Dense(classes, activation='softmax', use_bias=False)(fw_cell)
model = Model(inputs=x, outputs=h3)
start=timer.perf_counter()
model.compile(optimizer='Adam', loss='categorical_crossentropy')
end=timer.perf_counter()
print('>>> Model compilation took {:.1f} seconds'.format(end - start))

# Print parameter count
params = model.count_params()
print('# network parameters: ' + str(params))

# Check for correct sizes
assert (model.layers[-1].input_shape == (None, rnn_size))  # final projection input size (rnn_size)
assert (model.layers[-1].get_weights()[0].shape == (rnn_size, classes))  # final projection output size (rnn_size, classes)
output = model.predict(bX)
assert (output.shape == (batch_size, classes))

# Start training
batch_time = []
batch_loss = []
train_start=timer.perf_counter()
for i in range(batches):
    batch_start = timer.perf_counter()
    loss=model.train_on_batch(x=bX, y=to_categorical(bY, num_classes=classes))
    batch_end = timer.perf_counter()
    batch_time.append(batch_end - batch_start)
    batch_loss.append(loss)
train_end=timer.perf_counter()

# Write results
print_results(batch_time)
check_results(batch_loss, batch_time, train_start, train_end)
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=batch_time, version=version)
