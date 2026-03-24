import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Titanic=pd.read_csv("titanic.csv",sep='\t')
Titanic.head()
Titanic.shape
sb.heatmap(Titanic.isnull(),cmap="autumn")
Titanic.drop("Cabin",axis=1,inplace=True)
Titanic.head()
Titanic.dropna(inplace=True)
sb.heatmap(Titanic.isnull(),cmap="autumn")
gender=pd.get_dummies(Titanic['Sex']).head()
name=pd.get_dummies(Titanic['Name']).head()
pd.concat([gender,name],axis=1)