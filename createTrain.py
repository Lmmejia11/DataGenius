import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn import preprocessing
import numpy as np

#############  OPEN TRAIN ############

df1 = pd.read_csv('train.csv')

#############  ADD RANK AND POINTS ############

def merge_rank(df1 , df2  , player , y):
    df2['DATE_R']=pd.to_datetime(df2['DATE_R'])
    df1['DATE_G'] = pd.to_datetime(df1['DATE_G'])
    points =[]
    rank =[]
    i = 0
    for index, row in df1.iterrows():
        i = i+1
        if(i%500 == 0):
            print(i)
        joueur = row[player]
        selected = df2.loc[df2['ID_P_R'] == joueur]
        selected['diff']=(row['DATE_G'] - selected['DATE_R'])
        x = selected.loc[(selected['diff'])>timedelta(days=0)]
        try:
            a = x['diff'].idxmin()
            points.append(selected.get_value(index = a , col = 'POINT_R'))
            rank.append(selected.get_value(index = a , col = 'POS_R'))
        except:
            points.append(0)
            rank.append(901)
    df1['rank_'+y] = rank
    df1['points_'+y]= points
    return df1

df2 = pd.read_csv('player_rates.csv')
#df1 = merge_rank(df1,df2, 'ID1_G' ,'1')
#df1 = merge_rank(df1,df2, 'ID2_G' ,'2')
#df1.to_csv("ranking.csv")
df1 = pd.read_csv("ranking.csv", sep=";")

############ ADD STATS #####################################

df = pd.read_csv("stats.csv", sep=",")

dfp = df[['ID1','ID2','ID_T','ID_R']]
dfp['FS_1'] = df['FS_1']/df['FSOF_1']
dfp['ACES_1'] = df['ACES_1']/df['FSOF_1']
dfp['DF_1'] = df['DF_1']/df['FSOF_1']
dfp['W1S_1'] = df['W1S_1']/df['W1SOF_1']
dfp['W2S_1'] = df['W2S_1']/df['W2SOF_1']
dfp['BP_1'] =   df['BP_1']
dfp['FS_2'] = df['FS_2']/df['FSOF_2']
dfp['ACES_2'] = df['ACES_2']/df['FSOF_2']
dfp['DF_2'] = df['DF_2']/df['FSOF_2']
dfp['W1S_2'] = df['W1S_2']/df['W1SOF_2']
dfp['W2S_2'] = df['W2S_2']/df['W2SOF_2']
dfp['BP_2'] =   df['BP_2']
dfp.to_csv('stats2.csv')

# Merge with train
df1 = pd.merge(df1, dfp,  how='left', left_on=['ID1_G','ID2_G','ID_T_G','ID_R_G'], right_on = ['ID1','ID2','ID_T','ID_R'])
# Complete with mean where it isnt in stats
L1= ['ID1_G', "FS_1","ACES_1","DF_1","W1S_1","W2S_1","BP_1"]
L2= ['ID2_G', "FS_2","ACES_2","DF_2","W1S_2","W2S_2","BP_2"]
df1[L1[1:]] = df1[L1].groupby("ID1_G").transform(lambda x: x.fillna(x.mean()))
df1[L2[1:]] = df1[L2].groupby("ID2_G").transform(lambda x: x.fillna(x.mean()))
df1 = df1.fillna(df1.mean())

##########  ADD TOURNAMENT ###################################

#os.system("tour.py")
df2 =  pd.read_csv('tour2.csv')
df1  = df1.join(df2.set_index(['ID_T']) , on = 'ID_T_G' , rsuffix='_T')


############## SELECT AND SCALE #######################################

L = ['points_1','rank_1', 'FS_1' , 'ACES_1' , 'DF_1' , 'W1S_1', 'W2S_1' , 'BP_1','points_2','rank_2', 'FS_2' , 'ACES_2' ,
     'DF_2' , 'W1S_2', 'W2S_2' , 'BP_2' ,'RANK_T' , 'PRIZE_T' , 'LATITUDE_T','LONGITUDE_T','TYPE_T_1','TYPE_T_2','TYPE_T_3','TYPE_T_4','TYPE_T_5','TYPE_T_6','ID_R_G']
df1 = df1[L]

x = df1.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df1 = pd.DataFrame(x_scaled)
df1.columns = L

################# SYMETRIE ###############################

inv = df1.iloc[:,np.r_[8:16,0:8,16:27]]
inv['Resultat'] = 2
df1['Resultat'] = 1
inv.columns = df1.columns
df1 = df1.append(inv, ignore_index = True)

df1.to_csv('scaled_Train.csv.csv' , index = False)



