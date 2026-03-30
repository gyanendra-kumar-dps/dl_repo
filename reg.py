import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
from sklearn import datasets,linear_model
diabetes_x,diabetes_y=datasets.load_diabetes(return_X_y=True)
diabetes_x=diabetes_x[:,np.newaxis,2]
diabetes_x_train=diabetes_x[:-20]
diabetes_x_test=diabetes_x[-20:]
diabetes_y_train=diabetes_y[:-20]
diabetes_y_test=diabetes_y[-20:]
reg=linear_model.LinearRegression()
reg.fit(diabetes_x_train,diabetes_y_train)
dib_y_predict=reg.predict(diabetes_x_test)
print(reg.coef_)
print(mean_squared_error(diabetes_y_test,dib_y_predict))
print(r2_score(diabetes_y_test,dib_y_predict))
plt.scatter(diabetes_x_test,diabetes_y_test)
plt.plot(diabetes_x_test,dib_y_predict)
plt.show()