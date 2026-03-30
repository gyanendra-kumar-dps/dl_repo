import pandas as pd
import matplotlib.pyplot as plt
survived=pd.read_csv("survived.csv",sep='\t')
count_of_survived=survived["Survived"].value_counts()
count_of_survived.index=["Survived","Dead"]
plt.bar(count_of_survived.index,count_of_survived.values)
plt.show()