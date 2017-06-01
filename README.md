# rnn_benchmarks
Welcome to the rnn_benchmarks repository! We offer:
- A comparison of training speed of standard RNN implementations of the Lasagne, Tensorflow an pyTorch framework 
- Best-practice scripts on how to code up a network, optimizers, loss functions etc. when you are learning a new framework!

In order to run the benchmarks, you need the corresponding framework, then just run e.g. 'python gru_lsg.py'. The measured times will be written to 'results/results.csv'. After some runs, you can plot nice bar charts with 'main.py'.
The toy-data and default-parameters are provided by 'data.py', to make sure every script uses the same hyperparameters.

## Toy data
  - Toy batch of (25,1000,123) ~ (BatchSize, TimeSteps, Features)
  - Data sampled from uniform distribution between -1 and 1
  - Masking lengths: ([ 500,  520,  541, ..., 958,  979, 1000])
  - Cross Entropy labels: 25 classes
  - warp-ctc labels: TODO
  - 50 epochs of training

## Results on GTX 980 Ti
- Single layer unidirectional 320 GRU network ~0.42M params, Cross Entropy Loss on last output, no masking

  | Runtime per epoch [ms] | Lasagne | Tensorflow | PyTorch |
  |------------------------|---------|------------|---------|
  | Median                 | 34.3    | 59.0       | 16.3    |
  
- Quad layer 4x320 bidirectional LSTM network (EESEN) ~8.5M params, Cross Entropy Loss on final layers last output, masking

  | Runtime per epoch [ms] | Lasagne | Tensorflow | PyTorch |
  |------------------------|---------|------------|---------|
  | Median                 |  TODO     |    TODO      | TODO     |

- Quad layer 4x320 bidirectional LSTM network (EESEN) ~8.5M params, warp-ctc loss, masking

  | Runtime per epoch [ms] | Lasagne | Tensorflow | PyTorch |
  |------------------------|---------|------------|---------|
  | Median                 |  TODO     |    TODO      | TODO     |


## Gimme those bar charts
<img align="middle" src="/320-GRU/output.png" width="300">
