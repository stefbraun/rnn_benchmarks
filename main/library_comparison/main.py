import os
import subprocess
from pathlib import Path

# Parameters
cuda_device = 1
python_path = Path(__file__).resolve().parents[2]

# write path to dataframe to config file
with open(os.path.join(python_path, 'results', 'conf'), 'w') as f:
    f.write(os.path.join(python_path, 'results', 'library_comparison'))

# Please define your virtual environments for testing
lasagne = 'MKL_THREADING_LAYER=GNU LIBRARY_PATH=/usr/local/cuda-9.0/lib64 /home/brauns/anaconda3/envs/theano/bin/python'
keras_theano = 'LIBRARY_PATH=/usr/local/cuda-9.0/lib64 /home/brauns/anaconda3/envs/theano/bin/python'
tensorflow = '/home/brauns/anaconda3/envs/tensorflow/bin/python'
keras_tensorflow = '/home/brauns/anaconda3/envs/tensorflow/bin/python'
pytorch = '/home/brauns/anaconda3/envs/pt4/bin/python'

# 1x320 GRU bench
for interpreter_path, bench in zip([tensorflow, pytorch, lasagne], ['tensorflow', 'pytorch', 'lasagne']):

    script_path=os.path.join(python_path, '1x320-GRU', 'bench_{}.py'.format(bench))
    command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(cuda_device, python_path, interpreter_path,
                                                                                      script_path)
    print(command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 1x320 LSTM bench
for interpreter_path, bench in zip(
        [tensorflow, tensorflow, tensorflow, pytorch, pytorch, pytorch, lasagne, keras_tensorflow, keras_tensorflow, keras_theano],
        [ 'tensorflow_LSTMCell', 'tensorflow_LSTMBlockCell', 'tensorflow_cudnn', 'pytorch_cudnn', 'pytorch_cell',
         'pytorch_custom_cell', 'lasagne', 'keras_tensorflow', 'keras_tensorflow_cudnn', 'keras_theano']):
    if 'keras' in bench:
        backend = bench.split('_')[1]
        if 'cudnn' in bench:
            script_path = os.path.join(python_path, '1x320-LSTM', 'bench_keras_cudnn.py')
        else:
            script_path = os.path.join(python_path, '1x320-LSTM', 'bench_keras.py')
        command = 'MKL_THREADING_LAYER=GNU KERAS_BACKEND={} CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(backend, cuda_device, python_path, interpreter_path,
                                                                       script_path)
    else:
        script_path = os.path.join(python_path, '1x320-LSTM', 'bench_{}.py'.format(bench))
        command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(cuda_device, python_path, interpreter_path, script_path)
    print(command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 4x320 LSTM bench
for interpreter_path, bench in zip([tensorflow, tensorflow, pytorch, lasagne], ['tensorflow', 'tensorflow_cudnn', 'pytorch', 'lasagne']):
    script_path = os.path.join(python_path, '4x320-LSTM', 'bench_{}.py'.format(bench))
    command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(cuda_device, python_path, interpreter_path,
                                                                                       script_path)
    print(command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 4x320 LSTM bench, CTC
for interpreter_path, bench in zip([tensorflow, pytorch, lasagne], ['tensorflow', 'pytorch', 'lasagne']):
    script_path = os.path.join(python_path, '4x320-LSTM_ctc', 'bench_{}.py'.format(bench))
    command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(cuda_device, python_path,
                                                                                           interpreter_path, script_path)
    print(command)

    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()
#
