import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
df=pd.read_csv("insurance_data.csv")
plt.scatter(df["age"],df["bought_insurance"],color="red",marker="+")
X_train,X_test,y_train,y_test=train_test_split(df[["age"]],df["bought_insurance"],test_size=0.2)
model=LogisticRegression()
model.fit(X_train,y_train)
print("Model Coefficients:",model.coef_)
print("Model Intercept:",model.intercept_)
y_pred=model.predict(X_test)
print("Predicted values:",y_pred)
print("Actual values:",y_test.values)
score=model.score(X_test,y_test)
print("Model Accuracy:",score)
age=np.array([30,40,50]).reshape(-1,1)
predicted_probabilities=model.predict_proba(age)
print("Predicted probabilities for ages 30, 40, 50:\n",predicted_probabilities)
plt.scatter(df["age"],df["bought_insurance"],color="red",marker="+")
plt.plot(age,predicted_probabilities[:,1],color="blue")
plt.xlabel("Age")
plt.ylabel("Probability of Buying Insurance")
plt.title("Logistic Regression: Age vs Probability of Buying Insurance")
plt.show()