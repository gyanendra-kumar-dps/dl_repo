import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
r=pd.read_csv('ratings.csv')
r.head()
r.shape
r.describe()
r.isnull().sum()
r.dropna(inplace=True)
r.isnull().sum()
movies=pd.read_csv('movies.csv')
movies.head()
movies.shape
movies.describe()
movies.isnull().sum()
movies.dropna(inplace=True)
movies.isnull().sum()
data=pd.merge(r,movies,on='movieId')
data.head()
data.shape
data.describe()
data.isnull().sum()
data.dropna(inplace=True)
data.isnull().sum()
data['rating'].value_counts()
sb.countplot(data['rating'])
plt.show()
data['rating'].hist()
plt.show()
data['rating'].value_counts().plot(kind='bar')
plt.show()
data.groupby('title')['rating'].mean().sort_values(ascending=False).head()
data.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings=pd.DataFrame(data.groupby('title')['rating'].mean())
ratings.head()
ratings['num of ratings']=pd.DataFrame(data.groupby('title')['rating'].count())
ratings.head()
plt.figure(figsize=(10,4))