
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv('sample_data.csv')
df.head()
df.shape
df.describe()
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
plt.show()
X=df.drop('TARGET CLASS',axis=1)
y=df['TARGET CLASS']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
plt.figure(figsize=(8, 5))
plt.plot(k_values, errors, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error')
plt.title('K-Neighbors vs Error')
plt.grid(True)
plt.show()