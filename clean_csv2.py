import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# ds = pd.read_csv('./heart_failure.csv')

data = pd.read_csv('./heart.csv')
# print(np.array(data.keys()))
label=LabelEncoder()
object_col = data.select_dtypes(include='object')
non_object_col = data.select_dtypes(exclude='object')
for col in object_col.columns:
    object_col[col]=label.fit_transform(object_col[col])

df=pd.concat([object_col,non_object_col],axis=1)
items = []

# print(np.array(df.keys()))
# print(df.keys())
for row in np.array(df):
    if row[7] == 0:
        continue
    else:
        items.append(row)

# print(items)
dd = pd.DataFrame(data=items, columns=np.array(df.keys()))
dd.to_csv('heart_disease2.csv', index=False)