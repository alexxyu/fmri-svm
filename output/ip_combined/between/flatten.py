import sys
import numpy as np
import pandas as pd

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} INFILE OUTFILE")
    exit(1)

infile, outfile = sys.argv[1:]
data = np.load(infile)

subjects = ['AC', 'CD', 'CG', 'DA', 'JN', 'JS', 'SK']
k = len(subjects) - 1

df = pd.DataFrame(columns=['Subject', 'Block', 'Pre', 'Post'])
for b, block in enumerate(data):
    for i, subject in enumerate(subjects):
        for j in range(k):
            df.loc[len(df)] = [subject, b, block[1][i*k+j], block[0][i*k+j]]

#df.to_csv(outfile)
df.groupby(['Subject']).mean().to_csv(outfile)

#df_flattened = pd.DataFrame(columns=['Pre', 'Post'])
#df_flattened['Pre'] = df.groupby(['Subject'])['Pre'].mean()
#df_flattened['Post'] = df.groupby(['Subject'])['Post'].mean()
#df_flattened.to_csv(outfile)
