import torch
import torch.nn as nn
from torch.autograd import Variable
from support import toy_batch_ctc, default_params, write_results, print_results, plot_results
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from warpctc_pytorch import CTCLoss
import editdistance

# Experiment_type
framework = 'pytorch'
experiment = '4x320LSTM_CTC'

# Get data
bX, b_lenX, maskX, bY, b_lenY, classes = toy_batch_ctc()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, epochs = default_params()

# PyTorch compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))

# Create symbolic vars
bX = Variable(torch.from_numpy(bX).cuda())
bX = pack_padded_sequence(bX, b_lenX[::-1])  # Pack those sequences for masking, plz
b_lenX = Variable(torch.from_numpy(b_lenX))

bY = Variable(torch.from_numpy(bY))
b_lenY = Variable(torch.from_numpy(b_lenY))


# Create Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=inp_dims, hidden_size=rnn_size, num_layers=4, bias=True, bidirectional=True)
        self.fc = nn.Linear(rnn_size * 2, classes, bias=True)

    def forward(self, x):
        batch_size = x.batch_sizes[0]
        h1, state = self.lstm(x)
        h2, lens = pad_packed_sequence(h1)
        h3 = h2.view(-1, rnn_size*2)
        h4 = self.fc(h3)
        h5 = h4.view(-1, batch_size, classes)
        return h5


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

# Synchronize for more precise timing
torch.cuda.synchronize()

# Start training
time = []
for i in range(epochs):
    print('Epoch {}/{}'.format(i, epochs))
    start = timer()
    optimizer.zero_grad()
    output = net(bX)
    criterion = CTCLoss()  # loss definition
    loss = criterion(output, bY, b_lenX, b_lenY)
    loss.backward()
    optimizer.step()
    end = timer()
    time.append(end - start)
    output_numpy = output.cpu().data.numpy()
    assert (output_numpy.shape == (seq_len, batch_size, classes))


write_results(script_name=os.path.basename(__file__), framework=framework, experiment=experiment, parameters=params,
              run_time=time)
print_results(time)

# Plot results
fig, ax = plot_results(time)
fig.savefig('{}_{}.pdf'.format(framework, experiment), bbox_inches='tight')
