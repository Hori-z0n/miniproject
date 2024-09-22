# Discretization
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import lightgbm as gbm
import warnings 
warnings.filterwarnings('ignore')

data=pd.read_csv('./heart.csv')

cat=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

x=data.drop('HeartDisease',axis=1)
y=data['HeartDisease']

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.1, random_state = 1)
clf=LogisticRegression(max_iter=320, solver='liblinear')
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)

clf.score(x,y)
LOAC = metrics.accuracy_score(Y_test,y_pred)
LOPR = metrics.precision_score(Y_test,y_pred)
LORE = metrics.recall_score(Y_test,y_pred)
LOF1 = metrics.f1_score(Y_test,y_pred)

k=20
acc=np.zeros((k))
for i in range (1,k+1):
    nclf=KNeighborsClassifier(n_neighbors=i)
    nclf.fit(X_train,Y_train)
    ny_pred=nclf.predict(X_test)
    acc[i-1]=metrics.accuracy_score(Y_test,ny_pred)

nclf=KNeighborsClassifier(n_neighbors=9)
nclf.fit(X_train,Y_train)
ny_pred=nclf.predict(X_test)
KNAC = metrics.accuracy_score(Y_test,ny_pred)
KNPR = metrics.precision_score(Y_test,ny_pred)
KNRE = metrics.recall_score(Y_test,ny_pred)
KNF1 = metrics.f1_score(Y_test,ny_pred)

dtclf=DecisionTreeClassifier(max_depth=3)
dtclf.fit(X_train,Y_train)
dty_pred=dtclf.predict(X_test)
DTAC = metrics.accuracy_score(Y_test,dty_pred)
DTPR = metrics.precision_score(Y_test,dty_pred)
DTRE = metrics.recall_score(Y_test,dty_pred)
DTF1 = metrics.f1_score(Y_test,dty_pred)

rfclf=RandomForestClassifier(n_estimators=160,max_depth=5,random_state=110)
rfclf.fit(X_train,Y_train)
rfy_pred=rfclf.predict(X_test)
RFAC = metrics.accuracy_score(Y_test,rfy_pred)
RFPR = metrics.precision_score(Y_test,rfy_pred)
RFRE = metrics.recall_score(Y_test,rfy_pred)
RFF1 = metrics.f1_score(Y_test,rfy_pred)

Gclf=GaussianNB()
Gclf.fit(X_train,Y_train)
Gy_pred=Gclf.predict(X_test)
GAC = metrics.accuracy_score(Y_test,Gy_pred)
GPR = metrics.precision_score(Y_test,Gy_pred)
GRE = metrics.recall_score(Y_test,Gy_pred)
GF1 = metrics.f1_score(Y_test,Gy_pred)

svcclf=SVC(kernel='linear')
svcclf.fit(X_train,Y_train)
svcy_pred=svcclf.predict(X_test)
svcAC = metrics.accuracy_score(Y_test,svcy_pred)
svcPR = metrics.precision_score(Y_test,svcy_pred)
svcRE = metrics.recall_score(Y_test,svcy_pred)
svcF1 = metrics.f1_score(Y_test,svcy_pred)

lgb_model = gbm.LGBMClassifier()
lgb_model.fit(X_train, Y_train)
gbmy_pred = lgb_model.predict(X_test)
lgbAC = metrics.accuracy_score(Y_test, gbmy_pred)
lgbPR = metrics.precision_score(Y_test,y_pred)
lgbRE = metrics.recall_score(Y_test,y_pred)
lgbF1 = metrics.f1_score(Y_test,y_pred)

compar_accuracy=pd.DataFrame(columns=["algorithm","Accuracy"])
compar_accuracy.loc[len(compar_accuracy)] = ['LogisticRegression', LOAC]
compar_accuracy.loc[len(compar_accuracy)] = ['KNeighborsClassifier', KNAC]
compar_accuracy.loc[len(compar_accuracy)] = ['DecisionTreeClassifier', DTAC]
compar_accuracy.loc[len(compar_accuracy)] = ['RandomForestClassifier', RFAC]
compar_accuracy.loc[len(compar_accuracy)] = ['GaussianNB', GAC]
compar_accuracy.loc[len(compar_accuracy)] = ['svc', svcAC]
compar_accuracy.loc[len(compar_accuracy)] = ['lightgbm', lgbAC]

compar_precision=pd.DataFrame(columns=["algorithm","Precision"])
compar_precision.loc[len(compar_precision)] = ['LogisticRegression', LOPR]
compar_precision.loc[len(compar_precision)] = ['KNeighborsClassifier', KNPR]
compar_precision.loc[len(compar_precision)] = ['DecisionTreeClassifier', DTPR]
compar_precision.loc[len(compar_precision)] = ['RandomForestClassifier', RFPR]
compar_precision.loc[len(compar_precision)] = ['GaussianNB', GPR]
compar_precision.loc[len(compar_precision)] = ['svc', svcPR]
compar_precision.loc[len(compar_precision)] = ['lightgbm', lgbPR]

compar_recall=pd.DataFrame(columns=["algorithm","Recall"])
compar_recall.loc[len(compar_recall)] = ['LogisticRegression', LORE]
compar_recall.loc[len(compar_recall)] = ['KNeighborsClassifier', KNRE]
compar_recall.loc[len(compar_recall)] = ['DecisionTreeClassifier', DTRE]
compar_recall.loc[len(compar_recall)] = ['RandomForestClassifier', RFRE]
compar_recall.loc[len(compar_recall)] = ['GaussianNB', GRE]
compar_recall.loc[len(compar_recall)] = ['svc', svcRE]
compar_recall.loc[len(compar_recall)] = ['lightgbm', lgbRE]

compar_f1=pd.DataFrame(columns=["algorithm","F1"])
compar_f1.loc[len(compar_f1)] = ['LogisticRegression', LOF1]
compar_f1.loc[len(compar_f1)] = ['KNeighborsClassifier', KNF1]
compar_f1.loc[len(compar_f1)] = ['DecisionTreeClassifier', DTF1]
compar_f1.loc[len(compar_f1)] = ['RandomForestClassifier', RFF1]
compar_f1.loc[len(compar_f1)] = ['GaussianNB', GF1]
compar_f1.loc[len(compar_f1)] = ['svc', svcF1]
compar_f1.loc[len(compar_f1)] = ['lightgbm', lgbF1]

# sns.set()
fig, axes = plt.subplots(2, 2, figsize=(18, 5), sharey=True)

sns.barplot(ax=axes[0, 0], x='Accuracy', y='algorithm', data=compar_accuracy ,width=0.3,color='blue')
axes[0, 0].set_title('Accuracy')

sns.barplot(ax=axes[0, 1], x='Precision', y='algorithm', data=compar_precision ,width=0.3,color='blue')
axes[0, 1].set_title('Precision')

sns.barplot(ax=axes[1, 0], x='Recall', y='algorithm', data=compar_recall ,width=0.3,color='blue')
axes[1, 0].set_title('Recall')

sns.barplot(ax=axes[1, 1], x='F1', y='algorithm', data=compar_f1 ,width=0.3,color='blue')
axes[1, 1].set_title('F1')

print(compar_accuracy)
print('------------------------------------------------------------------------')
print(compar_precision)
print('------------------------------------------------------------------------')
print(compar_recall)
print('------------------------------------------------------------------------')
print(compar_f1)
plt.show()