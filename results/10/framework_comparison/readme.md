# Framework comparison
- PyTorch, TensorFlow, Lasagne, Keras/TensorFlow and Keras/Theano benchmarks
- LSTM implementations by cuDNN, fused kernels and naive approaches
- Fixed sequence-length data with cross-entropy loss and variable sequence-length data with CTC loss
- Input sizes 64x100x123 and 32x1000x123 (batch size x time steps x channels)
- Network sizes 1x320 and 4x320 (number of layers x number of LSTM units)

## Framework versions
Framework | Version | Release |Backend | cuda | cuDNN
-|-|-|-|-|-
PyTorch | 0.4.0 | [April 2018](https://github.com/PyTorch/PyTorch/releases/tag/v0.4.0) | - | 9.0 | 7102
TensorFlow | 1.8.0 | [April 2018](https://github.com/TensorFlow/TensorFlow/releases/tag/v1.8.0) |- | 9.0 | 7005 
Lasagne | 0.2.1dev | [April 2018 ](https://github.com/Lasagne/Lasagne/commit/7992faa80fa5233a786e2582a605e854cea7d1cf) | [Theano 1.0.1](https://github.com/Theano/Theano/releases/tag/rel-1.0.1) | 9.0 | 7005
Keras | 2.1.6 | [April 2018](https://github.com/Keras-team/Keras/releases/tag/2.1.6) |[Theano 1.0.1](https://github.com/Theano/Theano/releases/tag/rel-1.0.1), [TensorFlow 1.8.0](https://github.com/TensorFlow/TensorFlow/releases/tag/v1.8.0)| 9.0 | 7005

## LSTM implementations

Library | Name | Details
-|-|-
PyTorch | [`LSTMCell-basic`](https://github.com/stefbraun/rnn_benchmarks/blob/master/1x320-LSTM/lib_PyTorchLSTM.py) | Custom code, pure PyTorch implementation, easy to modify. Loop over time with Python `for` loop
PyTorch | [`LSTMCell-fused`](http://PyTorch.org/docs/stable/nn.html?highlight=lstmcell#torch.nn.LSTMCell) | LSTM with optimized kernel for single time steps. Loop over time with Python `for` loop
PyTorch |[`cuDNNLSTM`](http://PyTorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM) | Wrapper to cuDNN LSTM implementation
TensorFlow | [`LSTMCell`](https://www.TensorFlow.org/versions/r1.8/api_docs/python/tf/contrib/rnn/LSTMCell)| Pure TensorFlow implementation, easy to modify. Loop over time with `tf.while_loop`. Uses `dynamic_rnn`
TensorFlow | [`LSTMBlockCell`](https://www.TensorFlow.org/versions/r1.8/api_docs/python/tf/contrib/rnn/LSTMBlockCell)| Optimized LSTM with single operation per time-step. Loop over time with `tf.while_loop`. Uses `dynamic_rnn`
TensorFlow | [`LSTMBlockFusedCell`](https://www.TensorFlow.org/versions/r1.8/api_docs/python/tf/contrib/rnn/LSTMBlockFusedCell)| Optimized LSTM with single operation over all time steps. Loop over time is part of the operation.
TensorFlow | [`cuDNNLSTM`](https://www.TensorFlow.org/versions/r1.8/api_docs/python/tf/contrib/cuDNN_rnn/cuDNNLSTM)| Wrapper to cuDNN LSTM implementation
Lasagne | [`LSTMLayer`](http://Lasagne.readthedocs.io/en/latest/modules/layers/recurrent.html?highlight=gru#Lasagne.layers.LSTMLayer)| Pure Theano implementation, easy to modify. Loop over time with `theano.scan`
Keras | [`cuDNNLSTM`](https://Keras.io/layers/recurrent/#cuDNNlstm) | Wrapper to cuDNN LSTM implementation
Keras | [`LSTM`](https://Keras.io/layers/recurrent/#lstm)| Pure Theano/TensorFlow implementation, easy to modify. Loop over time with `theano.scan` or `tf.while_loop`

## Loss functions and input data
The loss functions are varied with the input data:
1. Cross-entropy for fixed sequence length data
  - default implementation from each framework
2. Connectionist Temporal Classification (CTC) for variable sequence length data
  - warp_ctc for [Theano+Lasagne](http://deeplearning.net/software/Theano/library/tensor/nnet/ctc.html?highlight=ctc#module-Theano.tensor.nnet.ctc) and [PyTorch](https://github.com/SeanNaren/warp-ctc)
  - TensorFlow default [CTC implementation](https://www.TensorFlow.org/api_docs/python/tf/nn/ctc_loss)


Benchmark name | Layers x LSTM units | # Classes or output units | Loss | Inputsize [NxTxC] <sup>1</sup> | Sequence length | Labels per sample| Comment
-|-|-|-|-|-|-|-
1x320/CE-short | 1x320 unidirectional | 10 Dense | cross entropy | 64x100x123 | fixed [100] | 1 |  Real world test<br>Dimensions are similar to isolated digit recognition on the TIDIGITS dataset<sup>2</sup>
1x320/CE-long | 1x320 unidirectional | 10 Dense | cross entropy | 32x1000x123 | fixed [1000] | 1 | Synthetic test
4x320/CE-long | 4x320 bidirectional | 10 Dense | cross entropy | 32x1000x123 | fixed [1000] | 1 | Synthetic test
4x320/CTC-long | 4x320 bidirectional | 59 Dense | CTC| 32x1000x123 | variable [500..1000] | 100 |  Real-world test <br>Dimensions are similar to continuous speech recognition on the WSJ dataset<sup>3
<!--
Loss | Input size [NxTxC]<sup>1 |  Sequence length |  Classes | Labels per sample | Comment
-|-|-|-|-
Cross-entropy | 64x100x123| fixed<br>[100] | 10 | 1 | Real world test<br>Dimensions are similar to isolated digit recognition on the TIDIGITS dataset<sup>1</sup>
Cross-entropy | 32x1000x123| fixed<br>[1000] | 10 | 1 | Synthetic test
CTC | 32x1000x123| variable<br>[500..1000] | 59 | 100 | Real-world test <br>Dimensions are similar to continuous speech recognition on the WSJ dataset<sup>2 -->
<sup>1</sup>N=number of samples, T=time-steps, C=feature channels<br>
<sup>2</sup>ASR-task on TIDIGITS/isolated digit recognition, default training set (0.7 hours of speech): 123-
dimensional filterbank features with 100fps, average sequence length of 98, alphabet size of 10 digits and
1 label per sample<br>
<sup>3</sup>ASR-task on WSJ/continuous speech recognition, pre-processing with [EESEN](https://github.com/srvk/eesen) on training subset
si-284 (81h of speech): 123-dimensional filterbank features with 100fps, average sequence length 783, alphabet
size of 59 characters and average number of characters per sample 102
## Results
- Xeon-W 2195 CPU, GTX 1080 Founders Edition, Ubuntu 16.04
- The results reflect the mean time to fully process a batch (forward + backward pass).
- The measurements are taken over 500 runs, and the first 100 are discarded as warm-up.

Benchmark | Results
-|-
1x320/CE-short<br>---<br>L1: 1x320 unidir LSTM<br>L2: 10 Dense<br>---<br>cross-entropy loss<br>input size 64x100x123<br>fixed sequence length<br>---<br>433k parameters<br> |<img align="middle" src="1x320-LSTM_cross-entropy_100.png" width="600">
1x320/CE-long<br>---<br>L1: 1x320 unidir LSTM<br>L2: 10 Dense<br>---<br>cross-entropy loss<br>input size 32x1000x123<br>fixed sequence length<br>---<br>576k parameters<br> | <img align="middle" src="1x320-LSTM_cross-entropy.png" width="600">
4x320/CE-long<br>---<br>L1-4: 4x320 bidir LSTM<br>L5: 10 Dense<br>---<br>cross-entropy loss<br>input size 32x1000x123<br>fixed sequence length<br>---<br>8.5M parameters<br> |<img align="middle" src="4x320-BIDIR-LSTM_cross-entropy.png" width="600">
4x320/CTC-long<br>L1-4: 4x320 bidir LSTM<br>L5: 59 Dense<br>---<br>CTC loss<br>input size 32x1000x123<br>variable sequence length<br>---<br>8.5M parameters<br> |<img align="middle" src="4x320-BIDIR-LSTM_CTC.png" width="600">

Remarks:
- The benchmark scripts are carefully written, but not optimized to squeeze that last bit of
performance out of them. They should reflect typical day-to-day research applications.
- Due to time constraints, only the 1x320 LSTM benchmark covers all considered frameworks.
For the multi-layer 4x320 networks, only implementations that provided helper functions to
create stacked bidirectional networks were evaluated. An exemption of this rule was made
for Lasagne, in order to include a Theano-based contender for this scenario.
- The TensorFlow benchmarks use the `feed_dict` input method that is simple to implement,
but slower than the [`tf.data` API](https://www.tensorflow.org/performance/performance_guide#input_pipeline_optimization). Implementing a high performance input pipeline in TensorFlow is not trivial, and only the feed_dict approach allowed for a similar implementation complexity as in the PyTorch and Lasagne cases.
- The TensorFlow `cuDNNLSTM` was not tested with variable length data as it does not support
such input (see [issue 6633](https://github.com/TensorFlow/TensorFlow/issues/6633)).
- The TensorFlow benchmark uses the integrated `tf.nn.ctc_loss` instead of the warp-
ctc library, even though there is a TensorFlow binding available ([Link](https://github.com/baidu-research/warp-ctc)). The performance
difference has not been measured.
- PyTorch 0.4.0 merged the Tensor and Variable classes and does not need the Variable
wrapper anymore. The Variable wrapper has a negligible performance impact on version
0.4.0, but is required for older PyTorch releases in the PyTorch version comparison.
- The CTC benchmark does not include the TensorFlow `cuDNNLSTM` implementation because it does not seem to handle variable sequence lengths (see [issue 6633](https://github.com/TensorFlow/TensorFlow/issues/6633)).
- The CTC benchmark was not carried out on PyTorch 0.1.12_2 as the compilation process was too complex. The packed sequence implementation has a large impact on performance for v0.2.0_4 (see [issue 4512](https://github.com/PyTorch/PyTorch/pull/4512)).
