import os
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from support import toy_batch, default_params, write_results, print_results, plot_results

# Experiment_type
framework = 'pytorch'
experiment = '1x320GRU'

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
        self.gru = nn.GRU(input_size=inp_dims, hidden_size=rnn_size, bias=True)
        self.fc = nn.Linear(rnn_size, classes, bias=True)

    def forward(self, x):
        h1, state = self.gru(x)  # RNN
        h2 = h1[-1, :, :]  # slice final output
        h3 = F.relu(self.fc(h2))
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

# Synchronize for more precise timing
torch.cuda.synchronize()

# Start training
time = []
ed = []
for i in range(epochs):
    print('Epoch {}/{}'.format(i, epochs))
    start = timer()
    optimizer.zero_grad()
    output = net(bX)
    criterion = nn.CrossEntropyLoss()  # loss definition
    loss = criterion(output, bY.long())
    loss.backward()
    optimizer.step()
    end = timer()
    time.append(end - start)
    output_numpy = output.cpu().data.numpy()

    # Test output size
    assert (output_numpy.shape == (batch_size, classes))

    # Test classification quality
    # target = bY.cpu().data.numpy()
    # prediction = np.argmax(output_numpy, axis=1)
    # ed.append(editdistance.eval(target, prediction))
    # print(ed[-1])

write_results(script_name=os.path.basename(__file__), framework=framework, experiment=experiment, parameters=params,
              run_time=time)
print_results(time)

# Plot results
fig, ax = plot_results(time)
fig.savefig('{}_{}.pdf'.format(framework, experiment), bbox_inches='tight')
