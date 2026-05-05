import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("water_potability.csv")

print(df.head())
print(df.isnull().sum())
df = df.fillna(df.mean(numeric_only=True))

X = df.drop("Potability", axis=1)

y = df["Potability"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
new_sample = [[
    7.0,    
    200,     
    15000,  
    7.0,    
    330,   
    400, 
    14,
    70,
    4 
]]

new_sample = scaler.transform(new_sample)

prediction = model.predict(new_sample)

if prediction[0] == 1:
    print("Water is Safe to Drink")
else:
    print("Water is Not Safe to Drink")
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Water Quality Correlation Heatmap")
plt.show()