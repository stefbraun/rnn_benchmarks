# rnn_benchmarks
Welcome to the rnn_benchmarks repository! We offer:
- A training speed comparison of standard RNN implementations of the Lasagne, Tensorflow an pyTorch framework 
- Best-practice scripts on how to code up a network, optimizers, loss functions etc. when you are learning a new framework!

In order to run the benchmarks, run the script 'main.py'. You need to give the paths to the python environment that contains the corresponding framework. The measured execution times will be written to 'results/results.csv'. 
The toy data and default parameters are provided by 'data.py', to make sure every script uses the same hyperparameters.

## Toy data
### Cross entropy loss function
  - Toy batch of (25,1000,123) ~ (BatchSize, TimeSteps, Features)
  - Uniform sequence length of 1000
  - Data sampled from uniform distribution between -1 and 1
  - 320 classes

### CTC loss function
  - Toy batch of (25,1000,123) ~ (BatchSize, TimeSteps, Features)
  - Variable sequence lengths to test masking: ([ 500,  520,  541, ..., 958,  979, 1000])
  - 58 classes (as EESEN), 100 labels per sample (~WSJ si84)
  
## Results on GTX 980 Ti
We train for 50 epochs and report the median of the runtimes per epoch in [msec].

  | Network              | Loss          | Masking | Lasagne | Tensorflow     | PyTorch     |
  |----------------------|---------------|---------|---------|----------------|-------------|
  | 1x320 unidir GRU (1) | Cross-Entropy | No      | 344 msec   | 588 msec      | 120 msec   |
  | 4x320 bidir LSTM (2)  | Cross-Entropy | No      | 3384 msec   | 5077 msec      | 1056 msec   |
  | 4x320 bidir LSTM (3)  | CTC           | Yes     | TODO   | TODO      | TODO   |
  
  1. L1: 320 GRU, L2: 15 Dense, ~43K params, cross entropy loss, no masking
  2. L1-L4: 4x320 bidir LSTM, L5: 15 Dense, ~8.5M params, cross entropy loss, no masking
  3. L1-L4: 4x320 bidir LSTM, L5: 59 Dense (similar to EESEN), ~8.5M params, CTC loss, masking


## Gimme those bar charts
<img align="middle" src="/1x320-GRU/bars.png" width="300">
<img align="middle" src="/4x320-LSTM/bars.png" width="300">

