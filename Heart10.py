import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from keras import Sequential
# from keras.layers import Dense
from sklearn.metrics import  auc
import pickle
import joblib
# import tensorflow as tf
import time

data = pd.read_csv('../heart_disease.csv')
# data = pd.read_csv('./heart_failure.csv')
# data = pd.read_csv('./heart.csv')
X = data.drop('HeartDisease', axis=1)
# X = data[['Age','Oldpeak', 'Cholesterol']].values
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=12)
# Normalization
sc = StandardScaler()
X_sc = sc.fit_transform(X)

model = linear_model.LogisticRegression(max_iter=1000, random_state=42)

history = model.fit(X, y)
print(X_test)
print(y_test)
# Sex  ChestPainType  RestingECG  ExerciseAngina  ST_Slope  Age  RestingBP  Cholesterol  FastingBS  MaxHR  Oldpeak
X_test_1 = [[1,0,1,1,2,60,130,253,0,144.0,1.4]] # 1
X_test_2 = [[1,1,1,0,2,35,122,192,0,174,0.0]] # 0
# print(type(np.array(X_test)[0][0]))
# print(X_test.keys())
# for i in (X_test.keys()):
#     print(f"{type(i)}")

# score = model.score(X_test, y_test)
# print(model.predict(X.values[0]))
# print("Test score: {0:.2f} %".format(100 * score))
Ypredict = model.predict(X_test_2)
print(int(Ypredict))
y_pred = np.where(Ypredict>0.5, 1, 0)
print(y_pred)
# print(len(Ypredict))
# print(len(np.array(y_test)))

# Ytarget = np.array(y_test)
# tp,tn,fp,fn = 0,0,0,0
# for i in range(150):
#     if Ytarget[i] == 1 and Ypredict[i] == 1:
#         tp += 1
#     elif Ytarget[i] == 1 and Ypredict[i] == 0:
#         fn += 1
#     elif Ytarget[i] == 0 and Ypredict[i] == 1:
#         fp += 1
#     elif Ytarget[i] == 0 and Ypredict[i] == 0:
#         tn += 1

# print(f"true positive{tp} true negative{tn} false positive{fp} false negative{fn}")