import torch
import torch.nn as nn
from torch.autograd import Variable
from data import toy_batch, default_params, write_results, print_results, plot_results
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import os

# Get data
bX, bY, b_lenX, maskX = toy_batch()
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

    def forward(self, x):
        output, state = self.gru(x)
        output = output[-1, :, :]
        return output


net = Net()
net.cuda()  # move network to GPU

# Print parameter count
total_count = 0
for param in list(net.parameters()):
    sizes = 1
    for el in param.size():
        sizes = sizes * el
    total_count += sizes
print('# network parameters: ' + str(total_count))

# Create optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Synchronize for more precise timing
torch.cuda.synchronize()

# Start training
time = []
for i in range(epochs):
    start = timer()
    optimizer.zero_grad()
    output = net(bX)
    criterion = nn.CrossEntropyLoss()  # loss definition
    loss = criterion(output, bY.long())
    loss.backward()
    optimizer.step()
    end = timer()
    time.append(end - start)
write_results(os.path.basename(__file__), time)
print_results(time)

# Plot results
plot_results(time)
plt.show()

