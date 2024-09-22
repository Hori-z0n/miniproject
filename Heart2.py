import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import  auc
from sklearn.neural_network import MLPClassifier
# import tensorflow as tf
import time
import tempfile
import os

# model_dir = tempfile.gettempdir()
# os.chdir('C:/Users/konla/OneDrive/Desktop/Python')
# model_dir = os.getcwd()
# print(model_dir)
# version = 1
# export_path = os.path.join(model_dir, str(version))
# print('export_path = {}\n',format(export_path))

# Discretization

Index = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak','HeartDisease', 'ChestPainType = ATA', 'ChestPainType = NAP','ChestPainType = ASY', 'RestingECG = Normal', 'RestingECG = ST','RestingECG = LVH', 'ExerciseAngina = N', 'ExerciseAngina = Y','ST_Slope = Up', 'ST_Slope = Flat', 'Sex = M', 'Sex = F']
# index = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS','RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease']
data = pd.read_csv('./heart_failure2.csv')
# data = pd.read_csv('./heart_failure.csv')
# data = pd.read_csv('./heart.csv')
X = data.drop('HeartDisease', axis=1)
# X = data[['Age','Oldpeak', 'Cholesterol']].values
y = data['HeartDisease']
# print(ds)
# print(X.keys())
sc = StandardScaler()
X_sc = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_sc, y, train_size=0.8, random_state=12)
print(X_train)

# Build Model
model = Sequential()
model.add(Dense(256, input_dim=18, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Normalization

#Compile model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

start = time.time()
# history = model.fit(X_train, Y_train, epochs=1000, verbose=1, batch_size=32, validation_split=0.5, callbacks=[es])
history = model.fit(X_train, y_train, epochs=100, verbose=1, batch_size=32, validation_split=0.25)
end = time.time()
print("Training Time: {:.3f} secs".format(end-start))

#Evaluation
score = model.evaluate(X_test, y_test, verbose=0)
print("Loss", score[0])
print("Accuracy:", score[1])

#Testing
y_pred_prob = model.predict(X_test)
y_pred = np.where(y_pred_prob>0.5, 1, 0)
# print("Predictions:\n{}".format(y_pred))

df_hist = pd.DataFrame.from_dict(history.history)

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