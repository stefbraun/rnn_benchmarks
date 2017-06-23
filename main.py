import subprocess

# Parameters
cuda_device = 1

# Please define your virtual environments for testing
lasagne = 'LIBRARY_PATH=/usr/local/cuda-8.0/lib64 /home/stefbraun/envs/theano/bin/python'
tensorflow = '/home/stefbraun/envs/tf/bin/python'
pytorch = '/home/stefbraun/anaconda3/envs/torch/bin/python'

# 1x320 GRU bench
for path, framework in zip([tensorflow, pytorch, lasagne], ['tensorflow', 'pytorch', 'lasagne']):
    command = 'CUDA_VISIBLE_DEVICES={} {} 1x320-GRU/bench_{}.py'.format(cuda_device, path, framework)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 1x320 LSTM bench (pytorch only)
for path, framework in zip([tensorflow, pytorch, pytorch, pytorch,  lasagne], ['tensorflow','pytorch_cudnn', 'pytorch_cell', 'pytorch_custom_cell', 'lasagne']):
    command = 'CUDA_VISIBLE_DEVICES={} {} 1x320-LSTM_variants/bench_{}.py'.format(cuda_device, path, framework)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 4x320 LSTM bench
for path, framework in zip([tensorflow, pytorch, lasagne], ['tensorflow', 'pytorch', 'lasagne']):
    command = 'CUDA_VISIBLE_DEVICES={} {} 4x320-LSTM/bench_{}.py'.format(cuda_device, path, framework)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 4x320 LSTM bench, CTC
for path, framework in zip([tensorflow, pytorch, lasagne], ['tensorflow', 'pytorch', 'lasagne']):
    command = 'CUDA_VISIBLE_DEVICES={} {} 4x320-LSTM_ctc/bench_{}.py'.format(cuda_device, path, framework)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

    # # # 4x320 LSTM bench
    # for path, framework in zip([tensorflow], ['tensorflow']):
    #     command = 'CUDA_VISIBLE_DEVICES={} {} 4x320-LSTM/bench_{}.py'.format(cuda_device, path, framework)
    #     proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    #     proc.wait()
    #
    # # 4x320 LSTM bench, CTC
    # for path, framework in zip([tensorflow], ['tensorflow']):
    #     command = 'CUDA_VISIBLE_DEVICES={} {} 4x320-LSTM_ctc/bench_{}.py'.format(cuda_device, path, framework)
    #     proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    #     proc.wait()

    # # 4x320 LSTM bench
    # for path, framework in zip([pytorch], ['pytorch']):
    #     command = 'CUDA_VISIBLE_DEVICES={} {} 4x320-LSTM/bench_{}.py'.format(cuda_device, path, framework)
    #     proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    #     proc.wait()
    #
    # # 4x320 LSTM bench, CTC
    # for path, framework in zip([pytorch], ['pytorch']):
    #     command = 'CUDA_VISIBLE_DEVICES={} {} 4x320-LSTM_ctc/bench_{}.py'.format(cuda_device, path, framework)
    #     proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    #     proc.wait()

    # # 4x320 LSTM bench
    # for path, framework in zip([lasagne], ['lasagne']):
    #     command = 'CUDA_VISIBLE_DEVICES={} {} 4x320-LSTM/bench_{}.py'.format(cuda_device, path, framework)
    #     proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    #     proc.wait()
    #
    # # 4x320 LSTM bench, CTC
    # for path, framework in zip([lasagne], ['lasagne']):
    #     command = 'CUDA_VISIBLE_DEVICES={} {} 4x320-LSTM_ctc/bench_{}.py'.format(cuda_device, path, framework)
    #     proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    #     proc.wait()
