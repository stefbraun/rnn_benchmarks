from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

# Set seeds
np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)


# Provide default parameters
def default_params():
    rnn_size = 320
    learning_rate = 1e-3
    epochs = 25
    return rnn_size, learning_rate, epochs


# Provide a toy batch
def toy_batch(seed=11, shape=(25, 1000, 123), classes=10):
    batch_size, max_len, features = shape
    np.random.seed(seed)

    # Samples
    bX = np.float32(np.random.uniform(-1, 1, (shape)))
    b_lenX = np.int32(np.linspace(max_len, max_len, batch_size))
    # print('::: Lengths of samples in batch: {}'.format(b_lenX))

    # Targets
    bY = np.int32(np.random.randint(low=0, high=classes - 1, size=batch_size))

    return bX, b_lenX, bY, classes


# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, max_len, features = bX.shape
rnn_size, learning_rate, epochs = default_params()

# PyTorch compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))

# Create symbolic vars
bX = Variable(torch.from_numpy(bX).cuda())
bX = pack_padded_sequence(bX, b_lenX[::-1])  # Pack those sequences for masking, plz

bY = Variable(torch.from_numpy(bY).cuda())


# Network 1: slow use of packed sequences
class slow_packed(nn.Module):
    def __init__(self, features, rnn_size):
        super(slow_packed, self).__init__()
        self.gru = nn.GRU(input_size=features, hidden_size=rnn_size, bias=True)
        self.linear = nn.Linear(rnn_size, classes, bias=True)

    def forward(self, x):
        h1p, state = self.gru(x)  # RNN
        h1, lens = pad_packed_sequence(h1p)  # unpack
        max_len, batch_size, features = h1.size()  # get sizes for reshape
        h2 = h1.view(max_len * batch_size, -1)  # reshape
        h3 = self.linear(h2)  # linear transform
        h4 = h3.view(max_len, batch_size, -1)  # reshape
        h5 = h4[-1, :, :]  # slice last element
        return h5


# Network 2: fast use of packed sequences
class fast_packed(nn.Module):
    def __init__(self, features, rnn_size):
        super(fast_packed, self).__init__()
        self.gru = nn.GRU(input_size=features, hidden_size=rnn_size, bias=True)
        self.linear = nn.Linear(rnn_size, classes, bias=True)

    def forward(self, x):
        h1p, state = self.gru(x)  # RNN
        h3 = self.linear(h1p.data)  # linear transform
        h3p = PackedSequence(h3, h1p.batch_sizes)  # create PackedSequence (pytorch docs: Instances of this class should never be created manually. They are meant to be instantiated by functions like pack_padded_sequence().
        h4, lens = pad_packed_sequence(h3p)  # unpack
        h5 = h4[-1, :, :]  # slice last element
        return h5


########################################################
# Switch between fast and slow variant by commenting
# net = slow_packed(features=features, rnn_size=rnn_size)
net = fast_packed(features=features, rnn_size=rnn_size)
########################################################
net.cuda()  # move network to GPU
print(net)

# Print parameter count
params = 0
for param in list(net.parameters()):
    sizes = 1
    for el in param.size():
        sizes = sizes * el
    params += sizes
print('::: # network parameters: ' + str(params))

# Create optimizer, criterion
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  # loss definition

# Start training
time = []
for ep in range(epochs):

    start = timer()
    optimizer.zero_grad()
    output = net(bX)
    loss = criterion(output, bY.long())
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()  # Synchronize for more precise timing
    end = timer()
    time.append(end - start)

    if ep % 5 == 0:
        print('::: Checksum of final layer output from epoch {}:{}'.format(ep, torch.sum(output).cpu().data.numpy()[0]))

print('>>> Median runtime per epoch [sec] {}'.format(np.median(time)))