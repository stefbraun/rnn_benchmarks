import os
import subprocess

# Parameters
cuda_device = 1

# Please define your virtual environments for testing
python_path = os.path.dirname(os.path.realpath(__file__))
print(python_path)

lasagne = 'LIBRARY_PATH=/usr/local/cuda-8.0/lib64 /home/stefbraun/envs/theano/bin/python'
keras_theano = 'LIBRARY_PATH=/usr/local/cuda-8.0/lib64 /home/stefbraun/envs/theano/bin/python'
tensorflow = '/home/stefbraun/envs/tf/bin/python'
keras_tensorflow = '/home/stefbraun/envs/tf/bin/python'
pytorch = '/home/stefbraun/anaconda3/envs/torch/bin/python'

# 1x320 GRU bench
for path, framework in zip([tensorflow, pytorch, lasagne], ['tensorflow', 'pytorch', 'lasagne']):
    command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} 1x320-GRU/bench_{}.py'.format(cuda_device, python_path, path,
                                                                                      framework)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 1x320 LSTM variants bench
for path, framework in zip(
        [tensorflow, keras_tensorflow, pytorch, pytorch, pytorch, lasagne, keras_theano, keras_tensorflow],
        ['tensorflow_LSTMCell', 'tensorflow_LSTMBlockCell', 'pytorch_cudnn', 'pytorch_cell',
         'pytorch_custom_cell', 'lasagne', 'keras_theano', 'keras_tensorflow']):
    command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} 1x320-LSTM_variants/bench_{}.py'.format(cuda_device,
                                                                                                python_path, path,
                                                                                                framework)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 4x320 LSTM bench
for path, framework in zip([tensorflow, pytorch, lasagne], ['tensorflow', 'pytorch', 'lasagne']):
    command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} 4x320-LSTM/bench_{}.py'.format(cuda_device, python_path, path,
                                                                                       framework)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 4x320 LSTM bench, CTC
for path, framework in zip([tensorflow, pytorch, lasagne], ['tensorflow', 'pytorch', 'lasagne']):
    command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} 4x320-LSTM_ctc/bench_{}.py'.format(cuda_device, python_path,
                                                                                           path, framework)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()
