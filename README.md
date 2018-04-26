# This repo will soon be updated with pytorch 0.4, tensorflow 1.7 and theano 1.0 benchmarks! 

# rnn_benchmarks
Welcome to the rnn_benchmarks repository! We offer:
- A training speed comparison of standard GRU and LSTM implementations of the the theano/lasagne, tensorflow an pytorch frameworks 
- Best-practice scripts on how to code up a network, optimizers, loss functions etc. when you are learning a new framework!

## Dependencies
  - [Tensorflow-1.2](https://github.com/tensorflow/tensorflow/tree/v1.2.0)
  - [Pytorch-0.1.12+1572173](https://github.com/pytorch/pytorch/tree/1572173ca735f379794d0ac10412208bbc0605b3)
  - [warp-ctc with pytorch binding](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
  - [Theano-0.9](https://github.com/Theano/Theano/releases/tag/rel-0.9.0)
  - [warp-ctc with theano binding](https://github.com/sherjilozair/ctc)
  - [Lasagne (latest)](https://github.com/Lasagne/Lasagne),
  - [Keras-2.0.5](https://github.com/fchollet/keras/releases/tag/2.0.5)

## Run the benchmarks
Go to the script 'main.py'. Before running, you need to give the paths to the python environment that contain the corresponding framework. The measured execution times will be written to 'results/results.csv'. 
The toy data and default parameters are provided by 'support.py', to make sure every script uses the same hyperparameters.

## Toy data
### Cross entropy loss function
  - Toy batch of (25,1000,123) ~ (BatchSize, TimeSteps, Features)
  - Uniform sequence length of 1000
  - Data sampled from uniform distribution between -1 and 1
  - 25 classes

### CTC loss function
  - Toy batch of (25,1000,123) ~ (BatchSize, TimeSteps, Features)
  - Variable sequence lengths to test masking: ([ 500,  520,  541, ..., 958,  979, 1000])
  - 59 classes (similar to [EESEN](https://github.com/srvk/eesen)), 100 labels per sample (~WSJ si84)
  - **Attention**: due to the different bindings for ctc, the following configurations had to be used:
    - tensorflow with tf-ctc (probably **GPU?**)
    - pytorch with warp-ctc (**GPU**)
    - lasagne with warp-ctc (**CPU**)
  
## Results on GTX 980 Ti
We train for 500 epochs and report the median of the runtimes per epoch in [sec].

  | Network              | Loss          | Masking | Lasagne    | Tensorflow  | PyTorch    |
  |----------------------|---------------|---------|------------|-------------|------------|
  | 1x320 unidir GRU (1) | Cross-Entropy | No      | 0.32 sec   | 0.59 sec    | 0.12 sec   |
  | 4x320 bidir LSTM (2) | Cross-Entropy | No      | 3.38 sec   | 5.07 sec    | 1.05 sec   |
  | 4x320 bidir LSTM (3) | CTC           | Yes     | 4.60 sec   | 5.22 sec    | 1.35 sec   |
  
  1. L1: 1x320 GRU, L2: 25 Dense, ~435K params, cross entropy loss, no masking
  2. L1-L4: 4x320 bidir LSTM, L5: 15 Dense, ~8.5M params, cross entropy loss, no masking
  3. L1-L4: 4x320 bidir LSTM, L5: 59 Dense, ~8.5M params, CTC loss, masking
  
  
  | Network              | Loss          | Masking | Lasagne    | Tensorflow <br> LSTMCell|Tensorflow <br> LSTMBlockCell| PyTorch<br>cudnn<br>LSTM | PyTorch<br>LSTMCell| Pytorch<br>Custom<br>LSTMCell|Keras <br> LSTM <br> backend theano | Keras <br> LSTM <br> backend tensorflow
  |----------------------|---------------|----|----------|----------|----------|----------|----------|----------|----------|-----|
  | 1x320 unidir LSTM (4)| Cross-Entropy | No | 0.40 sec | 0.59 sec | 0.52 sec | 0.18 sec | 0.44 sec | 1.16 sec | 0.72 sec | 0.83 sec|
  
  
  4. L1: 1x320 LSTM, L2: 25 Dense, ~578K params, cross entropy loss, no masking. See documentation: [PyTorch cudnn LSTM](http://pytorch.org/docs/nn.html#lstm), [PyTorch LSTMCell](http://pytorch.org/docs/nn.html#lstmcell), [PyTorch Custom LSTMCell](https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py)



