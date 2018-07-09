import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser(description='Process results dataframe')
parser.add_argument('--file', default=None,
                    help='Dataframe to process.')
args = parser.parse_args()

# Load file
logfile = os.path.join(args.file)
df = pd.read_csv(logfile)

# assert (int(df.groupby(['experiment', 'bench', 'version']).count()['runtime'].unique()) == 500)
df['runtime'] = df['runtime'] * 1000

df=df.groupby(['experiment','bench']).tail(400)
df['mean'] = df.groupby(['experiment','bench']).transform('mean')['runtime']
df['std'] = df.groupby(['experiment','bench']).transform('std')['runtime']
df = df.sort_values(['mean'], ascending=True)
grp=df.groupby(['experiment','bench'], as_index=False).tail(1).round(1)
print(grp.to_string())
