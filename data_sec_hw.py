import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matches=pd.read_csv("matches.csv",sep='\t')
matches.head()
matches.shape
matches.dropna(inplace=True)
sb.heatmap(matches.isnull(),cmap="winter")
deliveries=pd.read_csv("deliveries.csv",sep='\t')
deliveries.head()
deliveries.shape
deliveries.dropna(inplace=True)
sb.heatmap(deliveries.isnull(),cmap="winter")
plt.show()