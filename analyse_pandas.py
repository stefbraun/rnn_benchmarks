import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np

# Load file
repo_path = os.path.dirname(os.path.realpath(__file__))
logfile = os.path.join(repo_path, 'results/results_1080.csv')
df = pd.read_csv(logfile)

# Parameters
experiments =  list(df['experiment'].unique())

print(df.groupby(['experiment', 'framework']).mean())

print(df.groupby(['experiment', 'framework']).std())
