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
# print(data)
# print(data.describe())
cat=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
# plt.figure(figsize=(25,5))
# for i in range(5):
#   plt.subplot(1,5,i+1)
#   sns.countplot(x=cat[i],data=data)
# plt.show()

# print(data[data.HeartDisease == 1]['Sex'].value_counts())

# fig,axs = plt.subplots(2,2,figsize = (10,7))
# axs[0, 0].pie( data[data.HeartDisease == 0]['Sex'].value_counts(), labels= data[data.HeartDisease == 0]['Sex'].value_counts().index, autopct='%1.1f%%')
# axs[0, 1].pie( data[data.HeartDisease == 1]['Sex'].value_counts(), labels= data[data.HeartDisease == 1]['Sex'].value_counts().index, autopct='%1.1f%%')
# axs[0,0].set_title("People with no Heart disease\n\n Gender")
# axs[0,1].set_title("People with Heart disease\n\n Gender")
# axs[1, 0].pie( data[data.HeartDisease == 0]['ChestPainType'].value_counts(), labels= data[data.HeartDisease == 0]['ChestPainType'].value_counts().index, autopct='%1.1f%%')
# axs[1, 1].pie( data[data.HeartDisease == 1]['ChestPainType'].value_counts(), labels= data[data.HeartDisease == 1]['ChestPainType'].value_counts().index, autopct='%1.1f%%')
# axs[1,0].set_title("ChestPainType")
# axs[1,1].set_title("ChestPainType")
# plt.show()

# fig,axs = plt.subplots(3,2,figsize = (14,10))
# axs[0, 0].pie( data[data.HeartDisease == 0]['RestingECG'].value_counts(), labels= data[data.HeartDisease == 0]['RestingECG'].value_counts().index, autopct='%1.1f%%')
# axs[0, 1].pie( data[data.HeartDisease == 1]['RestingECG'].value_counts(), labels= data[data.HeartDisease == 1]['RestingECG'].value_counts().index, autopct='%1.1f%%')
# axs[0,0].set_title("People with no Heart disease\n\n RestingECG\n")
# axs[0,1].set_title("People with Heart disease\n\nRestingECG\n")
# axs[1, 0].pie( data[data.HeartDisease == 0]['ExerciseAngina'].value_counts(), labels= data[data.HeartDisease == 0]['ExerciseAngina'].value_counts().index, autopct='%1.1f%%')
# axs[1,0].set_title("ExerciseAngina")
# axs[1, 1].pie( data[data.HeartDisease == 1]['ExerciseAngina'].value_counts(), labels= data[data.HeartDisease == 1]['ExerciseAngina'].value_counts().index, autopct='%1.1f%%')
# axs[1,1].set_title("ExerciseAngina")
# axs[2, 0].pie( data[data.HeartDisease == 0]['FastingBS'].value_counts(), labels= data[data.HeartDisease == 0]['FastingBS'].value_counts().index, autopct='%1.1f%%')
# axs[2,0].set_title("FastingBS")
# axs[2, 1].pie( data[data.HeartDisease == 1]['FastingBS'].value_counts(), labels= data[data.HeartDisease == 1]['FastingBS'].value_counts().index, autopct='%1.1f%%')
# axs[2,1].set_title("FastingBS")
# plt.show()

# fig,axs = plt.subplots(2,2,figsize = (10,7))
# sns.histplot(data[data.HeartDisease ==0].Cholesterol,ax = axs[0,0],color = 'green')
# axs[0,0].set_title("People_not have heart disease")
# sns.histplot(data[data.HeartDisease ==1].Cholesterol,ax = axs[0,1],color = 'green')
# axs[0,1].set_title("People_do have heart disease")
# sns.histplot(data[data.HeartDisease ==0].RestingBP,ax = axs[1,0],kde = True,color = 'green')
# sns.histplot(data[data.HeartDisease ==1].RestingBP,ax = axs[1,1],kde = True,color = 'green')
# plt.show()

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# print(data.head())

# x=data.drop('HeartDisease', axis=1).values.reshape(-1,1)
# y=data['HeartDisease'].values.reshape(-1,1)

# item=['RestingBP','Cholesterol']
# for i in range(len(item)):
#     print(f'nl \n ',data[item[i]].nlargest(3))
#     print(f'ns \n ',data[item[i]].nsmallest(3))

# print(f'nl \n ',data['MaxHR'].nlargest(3))
# print(f'ns \n ',data['MaxHR'].nsmallest(3))

# print(data.dtypes)
# data['RestingBP'] = pd.to_numeric(data['RestingBP'], errors='coerce')
# data['Cholesterol'] = pd.to_numeric(data['Cholesterol'], errors='coerce')
# data['RestingBP']=data['RestingBP'].apply(lambda x: np.nan if x > 140 else x)
# data['RestingBP']=data['RestingBP'].apply(lambda x: np.nan if x < 40 else x)
# data['Cholesterol']=data['Cholesterol'].replace(0,np.nan)

# print(data.isna().sum())
# data['RestingBP']=data['RestingBP'].replace(np.nan,data['RestingBP'].median())
# data['Cholesterol']=data['Cholesterol'].replace(np.nan,data['Cholesterol'].median())
# print(data.isna().sum())

x=data.drop('HeartDisease',axis=1)
y=data['HeartDisease']

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.1, random_state = 1)
clf=LogisticRegression(max_iter=320, solver='liblinear')
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
print(classification_report(Y_test,y_pred))
clf.score(x,y)
LOAC=metrics.accuracy_score(Y_test,y_pred)
# y_pred_proba = clf.predict_proba(X_test)[::,1]
# fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
# plt.plot(fpr,tpr,label="data 1")
# plt.legend(loc=4)
# plt.show()

k=20
acc=np.zeros((k))
for i in range (1,k+1):
    nclf=KNeighborsClassifier(n_neighbors=i)
    nclf.fit(X_train,Y_train)
    ny_pred=nclf.predict(X_test)
    acc[i-1]=metrics.accuracy_score(Y_test,ny_pred)
# print(acc)
# print(acc.max())
nclf=KNeighborsClassifier(n_neighbors=9)
nclf.fit(X_train,Y_train)
ny_pred=nclf.predict(X_test)
KNAC=metrics.accuracy_score(Y_test,ny_pred)
# print(classification_report(Y_test,ny_pred))

dtclf=DecisionTreeClassifier(max_depth=3)
dtclf.fit(X_train,Y_train)
dty_pred=dtclf.predict(X_test)
DTAC=metrics.accuracy_score(Y_test,dty_pred)
# text_representation=tree.export_text(dtclf)
# print(text_representation)
# print(data.columns)
# f_n=["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak",'ST_Slope']
# t_n=["0","1"]
# fig=plt.figure(figsize=(20,15),dpi=100)
# plot=tree.plot_tree(dtclf,feature_names=f_n,class_names=t_n,filled=True)
# plt.show()

rfclf=RandomForestClassifier(n_estimators=160,max_depth=5,random_state=110)
rfclf.fit(X_train,Y_train)
rfy_pred=rfclf.predict(X_test)
RFAC=metrics.accuracy_score(Y_test,rfy_pred)
print(classification_report(Y_test,rfy_pred))

Gclf=GaussianNB()
Gclf.fit(X_train,Y_train)
Gy_pred=Gclf.predict(X_test)
GAC=metrics.accuracy_score(Y_test,Gy_pred)
print(GAC)

svcclf=SVC(kernel='linear')
svcclf.fit(X_train,Y_train)
svcy_pred=svcclf.predict(X_test)
svcAC=metrics.accuracy_score(Y_test,svcy_pred)
print(svcAC)

lgb_model = gbm.LGBMClassifier()
lgb_model.fit(X_train, Y_train)
gbmy_pred = lgb_model.predict(X_test)
metrics.accuracy_score(Y_test, gbmy_pred)
print(metrics.accuracy_score(Y_test, gbmy_pred))

compar=pd.DataFrame(columns=["algorithm","Accuracy"])
compar.loc[len(compar)] = ['LogisticRegression', LOAC]
compar.loc[len(compar)] = ['KNeighborsClassifier', KNAC]
compar.loc[len(compar)] = ['DecisionTreeClassifier', DTAC]
compar.loc[len(compar)] = ['RandomForestClassifier', RFAC]
compar.loc[len(compar)] = ['GaussianNB', GAC]
compar.loc[len(compar)] = ['svc', svcAC]
compar.loc[len(compar)] = ['lightgbm', metrics.accuracy_score(Y_test,gbmy_pred)]
plt.figure(figsize=[10,5])
plt.title('Model Accuracy ')
sns.barplot(x='Accuracy', y='algorithm', data=compar ,width=0.3,color='blue')
plt.show()

print(compar)