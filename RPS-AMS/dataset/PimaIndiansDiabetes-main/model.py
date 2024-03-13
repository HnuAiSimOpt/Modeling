# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:47:26 2021

@author: krishna reddy
"""
import pandas as pd
import numpy as np
import sklearn
import pickle

data=pd.read_csv('pima-indians-diabetes.csv')
data.head()

data.shape

data.info()

# All the dtatypes are as expected and we dont have any null values

data.describe()

from sklearn.preprocessing import MinMaxScaler,StandardScaler

mm=MinMaxScaler()
ss=StandardScaler()

y=data['Class']
y.head()

x=data.iloc[:,:-2]
x

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

x_train.shape

x_test.shape

lr.fit(x_train,y_train)



pickle.dump(lr,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[6,0.74,0.59,0.35,0.0,0.5,0.23,50]]))

'''

y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score

confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))

accuracy_score(y_test,y_pred)

# precison
precision_score(y_test,y_pred)

recall_score(y_test,y_pred)

x

# Predict on the new data

output=lr.predict([[6,0.74,0.59,0.35,0.0,0.5,0.23,50]])
output 
'''