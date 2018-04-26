import os
import time as timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from support import toy_batch, default_params, write_results, print_results, plot_results

# Experiment_type
bench = 'pytorch_cudnnGRU'
version = torch.__version__
experiment = '1x320-GRU_cross-entropy'

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
        self.gru = nn.GRU(input_size=inp_dims, hidden_size=rnn_size, bias=True)
        self.fc = nn.Linear(rnn_size, classes, bias=True)

    def forward(self, x):
        h1, state = self.gru(x)  # RNN
        h2 = h1[-1, :, :]  # slice final output
        h3 = self.fc(h2)
        return h3


net = Net()
net.cuda()  # move network to GPU

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

# Synchronize for more precise timing
torch.cuda.synchronize()

# Start training
time = []
ed = []
for i in range(batches):
    print('Batch {}/{}'.format(i, batches))

    torch.cuda.synchronize()
    start = timer.perf_counter()

    bXt = Variable(torch.from_numpy(bX).cuda())
    bYt = Variable(torch.from_numpy(bY).cuda())

    optimizer.zero_grad()
    output = net(bXt)
    loss = criterion(output, bYt.long())
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    end = timer.perf_counter()
    time.append(end - start)

    output_numpy = output.cpu().data.numpy()
    assert (output_numpy.shape == (batch_size, classes))

# Write results
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=time, version=version)
print_results(time)