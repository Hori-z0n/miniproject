import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from keras import Sequential
# from keras.layers import Dense
from sklearn.metrics import  auc
import pickle
import joblib
# import tensorflow as tf
import time

# Discretization

Index = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak','HeartDisease', 'ChestPainType = ATA', 'ChestPainType = NAP','ChestPainType = ASY', 'RestingECG = Normal', 'RestingECG = ST','RestingECG = LVH', 'ExerciseAngina = N', 'ExerciseAngina = Y','ST_Slope = Up', 'ST_Slope = Flat', 'Sex = M', 'Sex = F']
# index = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS','RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease']
data = pd.read_csv('./heart_disease2.csv')
# data = pd.read_csv('./heart_failure.csv')
# data = pd.read_csv('./heart.csv')
X = data.drop('HeartDisease', axis=1)
# X = data[['Age','Oldpeak', 'Cholesterol']].values
y = data['HeartDisease']
# print(ds)
# print(X.keys())

# Normalization
sc = StandardScaler()
X_sc = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_sc, y, train_size=0.8, random_state=12)

# Build Model
model = linear_model.LogisticRegression(max_iter=1000, random_state=42)
# model = RandomForestClassifier(n_estimators=1000,max_depth=5,random_state=110)

#Compile model
# opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
# history = model.fit(X_train, Y_train, epochs=1000, verbose=1, batch_size=32, validation_split=0.5, callbacks=[es])
history = model.fit(X_train, y_train)
end = time.time()
print("Training Time: {:.3f} secs".format(end-start))

pkl_filename = './MLandDs/heart_failure.sav'
with open(pkl_filename, 'wb') as file:
    joblib.dump(model, file)

#Testing
y_pred_prob = model.predict(X_test)
y_pred = np.where(y_pred_prob>0.5, 1, 0)
# print("Predictions:\n{}".format(y_pred))

# df_hist = pd.DataFrame.from_dict(history.history)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
auc = metrics.roc_auc_score(y_test, y_pred_prob)
# auc_score = auc(fpr, tpr)
# print("AUC:", auc_score)
plt.plot(fpr, tpr,label="AUC = "+str(auc))
# plt.plot(fpr, tpr, label="ROC Curve (AUC=%0.2f)" % auc_score)
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")

plt.legend(loc=4)
plt.show()