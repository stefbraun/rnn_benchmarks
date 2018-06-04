import os
import time as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from support import toy_batch, default_params, write_results, print_results, check_results

# Experiment_type
bench = 'pytorch_LSTMCell-fused'
version = torch.__version__
experiment = '1x320-LSTM_cross-entropy'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# PyTorch compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))


# Create Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTMCell(input_size=inp_dims, hidden_size=rnn_size, bias=True)
        self.fc = nn.Linear(rnn_size, classes, bias=False)

    def forward(self, x):
        max_len, batch_size, features = x.size()
        h_lstm = Variable(torch.zeros(batch_size, rnn_size)).cuda()
        c_lstm = Variable(torch.zeros(batch_size, rnn_size)).cuda()

        output = []
        for i in range(max_len):
            h_lstm, c_lstm = self.lstm(x[i], (h_lstm, c_lstm))
            output.append(h_lstm)

        h1 = torch.stack(output)
        h2 = h1[-1, :, :]
        h3 = self.fc(h2)
        return h3


net = Net()
net.cuda()

# Print parameter count
params = 0
for param in list(net.parameters()):
    sizes = 1
    for el in param.size():
        sizes = sizes * el
    params += sizes
print('# network parameters: ' + str(params))

# Create optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  # loss definition

# Check for correct sizes
assert (net.fc.in_features == rnn_size)  # final projection input size (rnn_size)
assert (net.fc.weight.cpu().data.numpy().shape == (
classes, rnn_size))  # final projection output size (classes, rnn_size)
bXt = Variable(torch.from_numpy(bX).cuda())
torch.cuda.synchronize()
output = net(bXt)
output_numpy = output.data.cpu().numpy()
assert (output_numpy.shape == (batch_size, classes))

# Start training
batch_time = []
batch_loss = []
train_start = timer.perf_counter()
for i in range(batches):
    torch.cuda.synchronize() # synchronize function call for precise time measurement
    batch_start = timer.perf_counter()

    bXt = Variable(torch.from_numpy(bX).cuda())
    bYt = Variable(torch.from_numpy(bY).cuda())

    optimizer.zero_grad()
    output = net(bXt)
    loss = criterion(output, bYt.long())
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize() # synchronize function call for precise time measurement
    batch_end = timer.perf_counter()
    batch_time.append(batch_end - batch_start)
    batch_loss.append(float(loss.data.cpu().numpy()))
train_end = timer.perf_counter() # end of training

# Write results
print_results(batch_time)
check_results(batch_loss, batch_time, train_start, train_end)
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=batch_time, version=version)
