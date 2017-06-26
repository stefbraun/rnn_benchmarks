import os
from timeit import default_timer as timer

import bnlstm as bl
import editdistance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from support import toy_batch, default_params, write_results, print_results, plot_results

# Experiment_type
framework = 'pytorch_custom_LSTMcell'
experiment = '1x320LSTM'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, epochs = default_params()

# PyTorch compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))

# Create symbolic vars
bX = Variable(torch.from_numpy(bX).cuda())
bY = Variable(torch.from_numpy(bY).cuda())


# Create Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = bl.LSTMCell(input_size=inp_dims, hidden_size=rnn_size, use_bias=True)
        self.fc = nn.Linear(rnn_size, classes, bias=True)

    def forward(self, x):
        max_len, batch_size, features =x.size()
        h_lstm = Variable(torch.zeros(batch_size, rnn_size)).cuda()
        c_lstm = Variable(torch.zeros(batch_size, rnn_size)).cuda()
        output = []
        for i in range(max_len):
            h_lstm, c_lstm = self.lstm(x[i], (h_lstm, c_lstm))
            output.append(h_lstm)
        h1 = torch.stack(output)

        h2 = h1[-1, :, :]
        h3 = F.relu(self.fc(h2))
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

# Start training
time = []
ed=[]
for i in range(epochs):
    print('Epoch {}/{}'.format(i, epochs))
    start = timer()
    optimizer.zero_grad()
    output = net(bX)
    criterion = nn.CrossEntropyLoss()  # loss definition
    loss = criterion(output, bY.long())
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end = timer()
    time.append(end - start)
    output_numpy = output.cpu().data.numpy()
    assert (output_numpy.shape == (batch_size, classes))

write_results(script_name=os.path.basename(__file__), framework=framework, experiment=experiment, parameters=params, run_time=time)
print_results(time)

# Plot results
fig, ax = plot_results(time)
fig.savefig('{}_{}.pdf'.format(framework, experiment), bbox_inches='tight')