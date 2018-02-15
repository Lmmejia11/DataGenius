import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

depth = 10
cdrop = ['TYPE_T_1', 'TYPE_T_2', 'TYPE_T_3', 'TYPE_T_4', 'TYPE_T_5', 'TYPE_T_6']

df = pd.read_csv("scaled_Train.csv")
df = df.drop(cdrop,axis=1)
df = df.sample(frac=1)

pd.plotting.scatter_matrix(df, alpha=0.2, c = df['Resultat'].tolist())
plt.savefig('pairs.png')
