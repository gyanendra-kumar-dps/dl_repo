import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn import linear_model
import pandas as pd
data = pd.read_csv('xydataset.csv',sep='\t')
xdata = data['x'].values
ydata = data['y'].values
x_train = xdata[:-20]
x_test = xdata[-20:]
y_train = ydata[:-20]
y_test = ydata[-20:]
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print("Coefficient:", reg.coef_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
plt.scatter(x_test, y_test, color='black', label='Actual Data')
plt.plot(x_test, y_pred, color='blue', linewidth=2, label='Prediction Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()