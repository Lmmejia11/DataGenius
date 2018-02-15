import pandas as pd
from sklearn import preprocessing
import os

############ OPEN TEST #################

df1 = pd.read_csv('test.csv')

#############  ADD LATEST RANK AND POINTS ############

df  =  pd.read_csv('player_rates.csv')
df['DATE_R']=pd.to_datetime(df['DATE_R'])
df=df.loc[df.reset_index().groupby(['ID_P_R'])['DATE_R'].idxmax()]
df = df[['ID_P_R','POINT_R','POS_R']]
df1 =  df1.join(df.set_index('ID_P_R') , on = 'ID1_G' , rsuffix='_joueur1')
df1 =  df1.join(df.set_index('ID_P_R') , on = 'ID2_G' , rsuffix='_joueur2')
values = {'POINT_R': 0, 'POINT_R_joueur2': 0, 'POS_R': 901, 'POS_R_joueur2': 901}
df1 = df1.fillna(value=values)

#############  ADD AVERAGE STATS ############

os.system("avg.py")
df = pd.read_csv('avgstats.csv', sep=",")
df1 =  df1.join(df.set_index('ID1') , on = 'ID1_G' , rsuffix='_joueur1')
df1 =  df1.join(df.set_index('ID1') , on = 'ID2_G' , rsuffix='_joueur2')

##########  ADD TOURNAMENT ###################################

df = pd.read_csv('tour2.csv')
df1 =  df1.join(df.set_index('ID_T') , on = 'ID_T_G' , rsuffix='_T')

##########  SELECT AND SCALE ###################################

L = ['POINT_R','POS_R', 'FS_1' , 'ACES_1' , 'DF_1' , 'W1S_1', 'W2S_1' , 'BP_1','POINT_R_joueur2','POS_R_joueur2', 'FS_1_joueur2' , 'ACES_1_joueur2' ,
     'DF_1_joueur2' , 'W1S_1_joueur2', 'W2S_1_joueur2' , 'BP_1_joueur2' ,'RANK_T' , 'PRIZE_T' , 'LATITUDE_T','LONGITUDE_T','TYPE_T_1','TYPE_T_2','TYPE_T_3','TYPE_T_4','TYPE_T_5','TYPE_T_6','ID_R_G']
df1 = df1[L]

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled= min_max_scaler.fit_transform(df1.values)
df1 = pd.DataFrame(x_scaled)
df1.columns = L

df1.to_csv('scaled_Test.csv', index = False)

