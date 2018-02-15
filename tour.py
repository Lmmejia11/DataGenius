import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing

def pos(row , value):
    if( row['ID_C_T']== value):
        return 1
    return 0
def money(row):
    s= row['PRIZE_T']
    try:
        t = s.find('K')
        b =  True
        if( t ==-1 ):
            t= s.find('M')
            b  = False
        s =  s[1:t]
        if( b ):
            return float(s)
        return float(s)*1000
    except:
        return s
    
        
    
df1 = pd.read_csv('tour.csv')
hdr =  ['ID_T','ID_C_T' , 'RANK_T' , 'PRIZE_T' ,'LATITUDE_T' , 'LONGITUDE_T']
df = df1[hdr]

## C_T to boolean
df['TYPE_T_1']=df.apply(lambda row: pos (row,1) , axis =1)
df['TYPE_T_2']=df.apply(lambda row: pos (row,2) , axis =1)
df['TYPE_T_3']=df.apply(lambda row: pos (row,3) , axis =1)
df['TYPE_T_4']=df.apply(lambda row: pos (row,4) , axis =1)
df['TYPE_T_5']=df.apply(lambda row: pos (row,5) , axis =1)
df['TYPE_T_6']=df.apply(lambda row: pos (row,6) , axis =1)

## Translate K, M in prize
df['PRIZE_T']=df.apply(lambda row: money (row) , axis =1)
df.drop(columns='ID_C_T')

## Fill prize
df['PRIZE_T']=df['PRIZE_T'].convert_objects(convert_dates=False, convert_numeric=True, convert_timedeltas=False, copy=True)
dfp = df[['RANK_T','PRIZE_T']].groupby("RANK_T").mean()
df['PRIZE_T'] = dfp['PRIZE_T']

## Fill other columns
df = df.fillna(df.mean())

df.to_csv('tour2.csv', index=False)

