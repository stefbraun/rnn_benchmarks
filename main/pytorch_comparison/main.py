import os
import subprocess
import sys
from pathlib import Path

# Parameters
cuda_device = 1
python_path = Path(__file__).resolve().parents[2]
dry = 1

command_list = ['echo {} > {}'.format(os.path.join(python_path, 'results', 'pytorch_comparison'),
                                      os.path.join(python_path, 'results', 'conf'))]

# write path to dataframe to config file
with open(os.path.join(python_path, 'results', 'conf'), 'w') as f:
    f.write(os.path.join(python_path, 'results', 'pytorch_comparison'))

# Please define your virtual environments for testing
all_interpreters = ['/home/brauns/anaconda3/envs/pt{}/bin/python'.format(i) for i in range(1, 5)]

# Experiments
all_experiments = ['1x320-LSTM', '4x320-LSTM', '4x320-LSTM_ctc']

# Run benches
for experiment in all_experiments:
    experiment_folder = os.path.join(python_path, experiment)
    all_benches = [script for script in os.listdir(experiment_folder) if ('bench' in script) and 'pytorch' in script]

    for interpreter in all_interpreters:
        for bench in all_benches:
            print('=' * 100)
            _, framework, cell = bench.split('_')

            interpreter_path = interpreter
            script_path = os.path.join(experiment_folder, bench)
            command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} {} {}'.format(cuda_device, python_path, interpreter_path,
                                                                           script_path)

            print(command)
            command_list.append(command)
            if dry == 0:
                proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                proc.wait()

command_list = map(lambda x: x + '\n', command_list)
with open(os.path.join(sys.path[0], 'commands.sh'), 'w') as f:
    f.writelines(command_list)
