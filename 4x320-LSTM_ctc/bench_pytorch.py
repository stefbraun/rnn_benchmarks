import os
import time as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from warpctc_pytorch import CTCLoss

from support import toy_batch_ctc, default_params, write_results, print_results, plot_results

# Experiment_type
bench = 'pytorch_cudnnLSTM'
version = torch.__version__
experiment = '4x320-BIDIR-LSTM_CTC'

# Get data
bX, b_lenX, maskX, bY, b_lenY, classes = toy_batch_ctc()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# PyTorch compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))

# Create Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=inp_dims, hidden_size=rnn_size, num_layers=4, bias=True, bidirectional=True)
        self.fc = nn.Linear(rnn_size * 2, classes, bias=True)

    def forward(self, x):
        h1p, state = self.lstm(x)
        h1, lens = pad_packed_sequence(h1p)
        h2 = self.fc(h1)
        return h2


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
criterion = CTCLoss()

# Synchronize for more precise timing
torch.cuda.synchronize()

# Start training
time = []
for i in range(batches):
    print('Batch {}/{}'.format(i, batches))

    torch.cuda.synchronize()
    start = timer.perf_counter()

    bXt = Variable(torch.from_numpy(bX).cuda())
    bXt = pack_padded_sequence(bXt, b_lenX[::-1])  # Pack those sequences for masking, plz
    b_lenXt = Variable(torch.from_numpy(b_lenX))
    bYt = Variable(torch.from_numpy(bY))
    b_lenYt = Variable(torch.from_numpy(b_lenY))

    optimizer.zero_grad()
    output = net(bXt)
    loss = criterion(output, bYt, b_lenXt, b_lenYt)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    end = timer.perf_counter()
    time.append(end - start)

    output_numpy = output.cpu().data.numpy()
    assert (output_numpy.shape == (seq_len, batch_size, classes))

# Write results
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=time, version=version)
print_results(time)
