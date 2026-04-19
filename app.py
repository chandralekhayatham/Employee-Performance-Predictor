import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Employee Performance Predictor")
st.write("This AI model predicts employee performance based on working hours, number of projects, and experience.")

# Load dataset
df = pd.read_csv("data.csv")

# Features & Target
X = df[['hours', 'projects', 'experience']]
y = df['performance']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# User inputs
st.subheader("Enter Employee Details")

hours = st.slider("Working Hours", 1, 12, 6)
projects = st.slider("Projects", 1, 10, 3)
experience = st.slider("Experience (years)", 0, 10, 2)

# Prediction
if st.button("Predict"):
    input_data = [[hours, projects, experience]]
    prediction = model.predict(input_data)

    st.subheader("Result")

    if prediction[0] == "High":
        st.success("High Performance 🚀")
    elif prediction[0] == "Medium":
        st.warning("Medium Performance ⚖️")
    else:
        st.error("Low Performance ⚠️")