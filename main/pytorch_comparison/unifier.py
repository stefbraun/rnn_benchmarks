import os
from pathlib import Path

import pandas as pd

repo_path = Path(__file__).resolve().parents[2]

# Get frames
df1 = pd.read_csv(os.path.join(repo_path, 'results', 'framework_comparison', 'results.csv'))
df2 = pd.read_csv(os.path.join(repo_path, 'results', 'pytorch_comparison', 'results.csv'))

# Get version in framework comparison
df1pt = df1[df1['bench'].str.contains('pytorch')]
pytorch_version = df1pt['version'].unique()[0]
print('Replacing pytorch version {}'.format(pytorch_version))

# Prepare pytorch comparison dataframe
df2pt = df2.copy()
df2pt.drop(df2pt[df2pt['version'] == pytorch_version].index, inplace=True)

# Prepare framework comparison dataframe
df2pt = df2pt.append(df1pt)
df2pt.reset_index

# save csv
df2pt.to_csv(os.path.join(repo_path, 'results', 'pytorch_comparison', 'results.csv'), index=None)
5 + 5
