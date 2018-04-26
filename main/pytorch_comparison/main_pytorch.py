import os
import subprocess
import itertools
from pathlib import Path

# Parameters
cuda_device = 1
python_path = Path(__file__).resolve().parents[2]

# write path to dataframe to config file
with open(os.path.join(python_path, 'results', 'conf'), 'w') as f:
    f.write(os.path.join(python_path, 'results', 'pytorch_comparison'))

# Please define your virtual environments for testing
pytorch1 = '/home/brauns/anaconda3/envs/pt1/bin/python'
pytorch2 = '/home/brauns/anaconda3/envs/pt2/bin/python'
pytorch3 = '/home/brauns/anaconda3/envs/pt3/bin/python'
pytorch4 = '/home/brauns/anaconda3/envs/pt4/bin/python'

# 1x320 GRU bench
for interpreter_path, bench in zip([pytorch1, pytorch2, pytorch3, pytorch4], ['pytorch']*4):

    script_path=os.path.join(python_path, '1x320-GRU', 'bench_{}.py'.format(bench))
    command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(cuda_device, python_path, interpreter_path,
                                                                                      script_path)
    print(command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 1x320 LSTM bench
all_interpreters=[pytorch1, pytorch2, pytorch3, pytorch4]
all_benches=['pytorch_cudnn', 'pytorch_cell', 'pytorch_custom_cell']
combos=list(itertools.product(all_interpreters, all_benches))
for interpreter_path, bench in combos:

    if bench=='keras_theano' or bench =='keras_tensorflow':
        backend = bench.split('_')[1]
        script_path = os.path.join(python_path, '1x320-LSTM', 'bench_keras.py'.format())
        command = 'MKL_THREADING_LAYER=GNU KERAS_BACKEND={} CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(backend, cuda_device, python_path, interpreter_path,
                                                                       script_path)
    else:
        script_path = os.path.join(python_path, '1x320-LSTM', 'bench_{}.py'.format(bench))
        command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(cuda_device, python_path, interpreter_path, script_path)
    print(command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 4x320 LSTM bench
for interpreter_path, bench in zip([pytorch1, pytorch2, pytorch3, pytorch4], ['pytorch']*4):

    script_path = os.path.join(python_path, '4x320-LSTM', 'bench_{}.py'.format(bench))
    command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(cuda_device, python_path, interpreter_path,
                                                                                       script_path)
    print(command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()

# 4x320 LSTM bench, CTC
for interpreter_path, bench in zip([pytorch1, pytorch2, pytorch3, pytorch4], ['pytorch']*4):

    script_path = os.path.join(python_path, '4x320-LSTM_ctc', 'bench_{}.py'.format(bench))
    command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(cuda_device, python_path, interpreter_path, script_path)
    print(command)

    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    proc.wait()
