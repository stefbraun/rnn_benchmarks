import os
import time as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from warpctc_pytorch import CTCLoss

from support import toy_batch_ctc, default_params, write_results, print_results, check_results

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
        self.fc = nn.Linear(rnn_size * 2, classes, bias=False)

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

# Check for correct sizes
assert (net.fc.in_features == 2*rnn_size)  # final projection input size (rnn_size)
assert (net.fc.weight.cpu().data.numpy().shape == (
classes, 2*rnn_size))  # final projection kernel size (classes, rnn_size)
bXt = Variable(torch.from_numpy(bX).cuda())
bXt = pack_padded_sequence(bXt, b_lenX[::-1])
torch.cuda.synchronize()
output = net(bXt)
output_numpy = output.data.cpu().numpy()
assert (output_numpy.shape == (seq_len, batch_size, classes))

# Start training
batch_time = []
batch_loss = []
train_start = timer.perf_counter()
for i in range(batches):
    torch.cuda.synchronize() # synchronize function call for precise time measurement
    batch_start = timer.perf_counter()

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
