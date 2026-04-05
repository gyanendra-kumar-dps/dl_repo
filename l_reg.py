import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
st.title("Petrol Consumption Prediction App")
uploaded_file = st.file_uploader("Upload petrol_consumption.csv", type=["csv"])
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(dataset.head())
    st.subheader("Dataset Description")
    st.write(dataset.describe())
    X = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
    y = dataset['Petrol_Consumption']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    st.subheader("Model Coefficients")
    st.write(coeff_df)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.subheader("Actual vs Predicted")
    st.write(df)
    st.subheader("Model Evaluation Metrics")
    st.write("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
    st.write("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    st.write("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
