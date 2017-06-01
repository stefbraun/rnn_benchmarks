import torch
import torch.nn as nn
from torch.autograd import Variable
from data import toy_batch, default_params, write_results, print_results
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


bX, bY, b_lenX, maskX = toy_batch()
batch_size, seq_len, inp_dims = bX.shape

# Torch compatibility
bX = np.transpose(bX, (1, 0, 2))
bX = Variable(torch.from_numpy(bX).cuda())
bY = Variable(torch.from_numpy(bY).cuda())

bX = pack_padded_sequence(bX,b_lenX[::-1]) # Pack those sequences for masking, plz

rnn_size, learning_rate, epochs = default_params()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=inp_dims, hidden_size=rnn_size, num_layers=4, bias=True, bidirectional=True)

    def forward(self, x, hidden, memory):
        output, state = self.lstm(x, (hidden, memory))
        output, lens = pad_packed_sequence(output)
        # print(lens)
        output = output[-1, :, :]
        return output


hidden = Variable(torch.zeros(8, batch_size, rnn_size).cuda())
memory = Variable(torch.zeros(8, batch_size, rnn_size).cuda())

net = Net()
net.cuda()

total_count = 0
for param in list(net.parameters()):
    sizes=1
    for el in param.size():
        sizes = sizes * el
    total_count += sizes
print(total_count)

# Count parameters
params = list(net.parameters())
# create optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

torch.cuda.synchronize()

# Training
time = []
for i in range(epochs):
    start = timer()
    optimizer.zero_grad()
    output = net(bX, hidden, memory)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, bY.long())
    loss.backward()
    optimizer.step()
    end = timer()
    time.append(end - start)

write_results(os.path.basename(__file__), time)
print_results(time)

print(output)
plt.scatter(range(len(time)), time)
plt.grid()
plt.show()