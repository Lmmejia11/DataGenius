from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

depth = 10
cdrop = ['TYPE_T_1', 'TYPE_T_2', 'TYPE_T_3', 'TYPE_T_4', 'TYPE_T_5', 'TYPE_T_6']

df = pd.read_csv("scaled_Train.csv")
df = df.drop(cdrop,axis=1)
df = df.sample(frac=1)
X = df.drop(["Resultat"],axis=1)
X = X.values
Y = df["Resultat"].values

n = len(X)//3
X1 = X[0:n]
Y1 = Y[0:n]
X2 = X[n:2*n]
Y2 = Y[n:2*n]
X3 = X[2*n:]
Y3 = Y[2*n:]

clf = RandomForestClassifier(max_depth=depth, random_state=0)

X_train = np.concatenate((X1, X2), axis=0)
Y_train = np.concatenate((Y1, Y2), axis=0)
X_test = X3
Y_test = Y3
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
p = sum(np.equal(Y_pred,Y_test))/len(Y_test)
print(p)

X_train = np.concatenate((X1, X3), axis=0)
Y_train = np.concatenate((Y1, Y3), axis=0)
X_test = X2
Y_test = Y2
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
p = sum(np.equal(Y_pred,Y_test))/len(Y_test)
print(p)

X_train = np.concatenate((X3, X2), axis=0)
Y_train = np.concatenate((Y3, Y2), axis=0)
X_test = X1
Y_test = Y1
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
p = sum(np.equal(Y_pred,Y_test))/len(Y_test)
print(p)
