import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=50,centers=2,cluster_std=0.6)
clf=SGDClassifier(loss='hinge',alpha=0.01,max_iter=200)
clf.fit(x,y)
xx=np.linspace(-1,5,10)
yy=np.linspace(-1,5,10)
x1,x2=np.meshgrid(xx,yy)
z=np.empty(x1.shape)
for (i,j),val in np.ndenumerate(xx,yy):
    X1=val
    X2=x2[i,j]
    p=clf.decision_function([[x1,x2]])
    z[i,j]=p[0]
lvl=[-1,0,1]
linestyles=['dashed','solid','dashed']
colors='k'
plt.contour
plt.scatter
plt.axis
plt.show()