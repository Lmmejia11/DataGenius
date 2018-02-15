import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

df = pd.read_csv("stats2.csv", sep=",")

inv = df.iloc[:,np.r_[1,0,2:5,11:17,5:11]]
print(df.columns)
print(inv.columns)
inv.columns = df.columns
df2 = df.append(inv, ignore_index = True)

dfp = df[['ID1','FS_1','ACES_1','DF_1','W1S_1' ,'W2S_1','BP_1']].groupby("ID1").mean()
dfp = dfp.fillna(dfp.mean())
dfp.to_csv("avgstats.csv")

