import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv('sample_data(1).csv')
df.head()
y=df.pop('TARGET CLASS')
x=df
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(accuracy_score(y_test,y_pred))