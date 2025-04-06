import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)  # 0 = Setosa, 1 = Versicolor, 2 = Virginica

# Train the model on all data
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Iris Flower Predictor")
st.write("Enter the flower's measurements to predict its species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.01)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.01)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.01)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.01)

if st.button("Predict"):
    try:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]
        predicted_class = int(round(prediction))  # Class prediction by rounding
        species = iris.target_names[predicted_class]

        st.success(f"Predicted Species: **{species.capitalize()}** ")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
