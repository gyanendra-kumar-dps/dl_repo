from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
iris=load_iris()
X=iris.data[:, :2]
y=iris.target
model=LogisticRegression()
model.fit(X,y)
print("Model Coefficients:",model.coef_)
print("Model Intercept:",model.intercept_)
y_pred=model.predict(X)
print("Predicted values:",y_pred)
print("Actual values:",y)
score=model.score(X,y)
print("Model Accuracy:",score)
plt.scatter(X[:,0],X[:,1],c=y,cmap="viridis")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Logistic Regression: Iris Dataset")
plt.show()