# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:24:06 2020

@author: Adithya Madhavan
"""

#python : Topic :Decision Tree using mtcars

#standard libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import seaborn as sns
df = data('mtcars')
df.head()
#from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
df['am'].value_counts()

#classification

#predict if transmission of car is 0 or 1 on basis of mpg, hp, wt
X1 = df[['mpg','hp','wt']]
Y1 = df[['am']]
X1
Y1
#Y1 = df.value_counts()
Y1['am'].value_counts()


#split data
#from sklearn.model_selection import train_test_split
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=.20)
X_train.shape
X_test.shape

#model
from sklearn.tree import DecisionTreeClassifier
clsModel = DecisionTreeClassifier()  #model with parameter
clsModel.fit(X_train, y_train)

#predict
ypred1 = clsModel.predict(X_test)
len(ypred1)
X_test.head(6)
ypred1
Y1
#metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
classification_report(y_true=y_test, y_pred= ypred1)
confusion_matrix(y_true=y_test, y_pred=ypred1)
accuracy_score(y_true=y_test, y_pred=ypred1)



#regression
#predict if mpg (numerical value) on basis of am, hp, wt
X2 = df[['am','hp','wt']]
Y2 = df[['mpg']]
Y2['mpg'].value_counts()

#split data
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=.20, random_state=123 )
X2_train.shape
X2_test.shape
#10/data2.shape[0]

#model
from sklearn.tree import DecisionTreeRegressor #note this
regrModel = DecisionTreeRegressor()  #model with parameter
regrModel.fit(X2_train, y2_train)

#predict
ypred2 = regrModel.predict(X2_test)
ypred2
len(ypred2)
type(ypred2)
type(y2_test)
df2= y2_test
df2
df2['ypred2'] = ypred2
df2
#y2_test['ypred2']=ypred2
#y2_test 

#metrics
from sklearn import metrics
#Mean Abs Error
metrics.mean_absolute_error(y_true=y2_test, y_pred=ypred2)
sum(abs(df2['diff']))/df2.shape[0]

#Mean Squared Error (MSE)
metrics.mean_squared_error(y_true=y2_test, y_pred=ypred2)

#Root Mean Squared Error (RMSE)
np.sqrt(metrics.mean_squared_error(y_true=y2_test, y_pred=ypred2))
np.sqrt(metrics.mean_squared_error(y_true=df2['mpg'].values, y_pred=df2['ypred2'].values))
