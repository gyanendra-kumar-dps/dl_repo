import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('Pokemon Data (1).csv')
data.head()
data.info()
data.isnull().sum()
data['Type 2'].fillna(value='None',inplace=True)
data.isnull().sum()
data['Type 1'].value_counts().plot.bar()
data['Type 2'].value_counts().plot.bar()
data['Legendary'].value_counts().plot.bar()
plt.show()
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data['Legendary'] = lb.fit_transform(data['Legendary'])
from sklearn.model_selection import train_test_split

y = data.pop('Legendary')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)