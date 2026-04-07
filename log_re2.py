from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
x,y=make_classification(n_samples=100,n_features=2,n_classes=2,n_informative=2,n_redundant=0,random_state=42)
model=LogisticRegression()
model.fit(x,y)
p_pred=model.predict_proba(x)
y_pred=model.predict(x)
score=model.score(x,y)
conf_m=confusion_matrix(y,y_pred)
report=classification_report(y,y_pred)
print("Predicted probabilities:\n",p_pred)
print("Predicted classes:\n",y_pred)
print("Accuracy score:",score)
print("Confusion Matrix:\n",conf_m)
print("Classification Report:\n",report)
plt.scatter(x[:,0],x[:,1],c=y_pred,cmap='bwr',edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('a')
plt.show()