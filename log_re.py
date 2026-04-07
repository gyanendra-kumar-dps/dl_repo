import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
x=np.arange(10).reshape(-1,1)
y=np.array([0,0,0,0,0,1,1,1,1,1])
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