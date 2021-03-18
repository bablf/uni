from scipy.stats import chisquare, chi2_contingency
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np

gpt = [[51394, 25962, 19242, 23334, 15928, 19953], # 800
    [54808, 23351, 18752, 21267, 13411, 20373], # 400
    [48257, 27360, 18009, 21127, 14416, 20653], # 200
    [48387, 28602, 14504, 21038, 11546, 17594], # 100
    [47712, 26492, 14153, 22817, 12232, 11959], # 50
    [12345, 67890, 12345, 67890, 12345, 67890]] # Sanity check


twic = [49386,
        31507,
        28263,
        31493,
        22818,
        23608]

for i in range(len(gpt)):
    z, p = chisquare(f_obs=gpt[i],f_exp=twic)
    print("model {}\n\tz = {}, p = {}".format(i, z, p))

    ar = np.array([gpt[i], twic])
    print("\n",ar)

    df = pd.DataFrame(ar, columns=['pawns', 'rooks', 'bishops', 'knights', 'queen', 'king'], index=['gpt-2', 'twic'])
    print("\n", df)

    z,p,dof,expected=chi2_contingency(df, correction=False)
    print("\n\tNo correction: z = {}, p = {}, DOF = {}, expected = {}".format(z, p, dof, expected))

    z,p,dof,expected=chi2_contingency(df, correction=True)
    print("\n\tCorrected: z = {}, p = {}, DOF = {}, expected = {}".format(z, p, dof, expected))

    cor = pearsonr(gpt[i], twic)
    print("\n\tCorrelation = {}".format(cor))

