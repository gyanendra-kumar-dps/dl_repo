from numpy import where
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
x,y=make_blobs(1000,centers=2,random_state=1)
count=Counter(y)
for i in range(1,10):
    print(x[i],y[i])
for i,_ in count.items():
  row_index=where(y==i)[0]
  plt.scatter(x[row_index,0],x[row_index,1],label=str(i))
plt.show()