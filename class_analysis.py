import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.pyplot as mp
from sklearn.datasets import load_breast_cancer
dataset=load_breast_cancer()
sb.set_style('dark')
mp.style.use(['https://gist.githubusercontent.com/BrendanMartin/01e71bb9550774e2ccff3af7574c0020/raw/6fa9681c7d0232d34c9271de9be150e584e606fe/lds_default.mplstyle'])
mp.rcParams.update({"figure.figsize":(0,6),"axes.titlepad":22.0})
(unique,counts)=np.unique(dataset['target'],return_counts=True)
sb.barplot(x=dataset['target_names'],y=counts)
plt.show()