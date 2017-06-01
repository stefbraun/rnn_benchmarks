# rnn_benchmarks
Welcome to the rnn_benchmarks repository! We offer:
- A comparison of training speed of standard RNN implementations of the Lasagne, Tensorflow an pyTorch framework 
- Best-practice scripts on how to code up a network, optimizers, loss functions etc. when you are learning a new framework!

In order to run the benchmarks, you need the corresponding framework, then just run e.g. 'python gru_lsg.py'. The measured times will be written to 'results/results.csv'. After some runs, you can plot nice bar charts with 'main.py'.
The toy-data and default-parameters are provided by 'data.py', to make sure every script uses the same hyperparameters.

Median runtimes for a single layer unidirectional 320 GRU network, Cross Entropy Loss 
<img src="/320-GRU/output.png" width="300">
