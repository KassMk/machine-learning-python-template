import streamlit as st
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "iris_model.pkl")
model = joblib.load(model_path)

st.title("Clasificador de Flores Iris (Streamlit)")

st.write("""
Esta aplicación predice la **especie de Iris** en base a medidas del sépalo y pétalo.
""")

st.sidebar.header("Introduce las características:")

sepal_length = st.sidebar.number_input("Longitud del sépalo", 4.0, 8.0, 5.0)
sepal_width  = st.sidebar.number_input("Ancho del sépalo", 2.0, 4.5, 3.0)
petal_length = st.sidebar.number_input("Longitud del pétalo", 1.0, 7.0, 4.0)
petal_width  = st.sidebar.number_input("Ancho del pétalo", 0.1, 2.5, 1.0)

if st.sidebar.button("Predecir"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    species = ["Setosa", "Versicolor", "Virginica"]

    st.subheader("Predicción:")
    st.success(f"La flor es: {species[prediction]}")