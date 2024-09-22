import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  ,GradientBoostingClassifier

from sklearn.metrics import confusion_matrix , classification_report

data=pd.read_csv("./heart.csv")
# print(data.head())
# print(data.tail())
# print(data.shape)
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())
# sns.heatmap(data.isnull())
# print(data.duplicated().sum())
# print(data['HeartDisease'].value_counts())
# data.hist(figsize=(15,10))
# plt.show()
label=LabelEncoder()
object_col = data.select_dtypes(include='object')
# print(object_col.head())
non_object_col = data.select_dtypes(exclude='object')
# print(non_object_col.head())

for col in object_col.columns:
    object_col[col]=label.fit_transform(object_col[col])
# print(object_col.head())


df=pd.concat([object_col,non_object_col],axis=1)

# print(df.head())
# print(df.shape)

# Drop  target column 'HeartDisease' 
x= df.drop(['HeartDisease'],axis=1)

# Assign  target column 'HeartDisease' to 'y'
y=df['HeartDisease']
x_train , x_test , y_train , y_test = train_test_split(x,y,train_size=0.8,random_state=42)

def compare_classifiers(x_train, x_test, y_train, y_test):
    # Initialize the classifiers
    classifiers = {
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Support Vector Classifier': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    
    # Iterate over the classifiers, train, and evaluate them
    for name, clf in classifiers.items():
        print(f'--- {name} ---')
        
        # Train the classifier
        clf.fit(x_train, y_train)
        
        # Make predictions on the test data
        y_pred = clf.predict(x_test)
        
        # Print confusion matrix and classification report
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))
        print('\n' + '-'*40 + '\n')

compare_classifiers(x_train, x_test, y_train, y_test)