from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

depth = 10
cdrop = ['TYPE_T_1', 'TYPE_T_2', 'TYPE_T_3', 'TYPE_T_4', 'TYPE_T_5', 'TYPE_T_6']

df = pd.read_csv("scaled_Train.csv", sep=";")
df = df.drop(cdrop,axis=1)
df = df.sample(frac=1)
X_train = df.drop(["Resultat"],axis=1)
X_train = X_train.values
Y_train = df["Resultat"].values

df = pd.read_csv("scaled_Test.csv")
df = df.drop(cdrop,axis=1)
X_test = df

# clf = svm.SVC()
clf = RandomForestClassifier(max_depth=depth, random_state=0)

clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

dft = pd.read_csv("test.csv")
dft['WINNER'] = pd.DataFrame(Y_pred)
dft.to_csv("results.csv")

