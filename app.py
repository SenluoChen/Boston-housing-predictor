import streamlit as st
import joblib
import pandas as pd

# Load model and feature names
model = joblib.load("models/model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

def predict_custom_house(next_to_river, nr_rooms, students_per_classroom, distance_to_town, pollution_level, poverty_level):
    # Map UI inputs to the features expected by the trained model
    input_data = pd.DataFrame([{
        'CRIM': 0.1,  # default value
        'ZN': 0,
        'INDUS': 5,
        'NOX': 0.4 if pollution_level == 'high' else 0.2,
        'RM': nr_rooms,
        'AGE': 30,
        'DIS': distance_to_town,
        'TAX': 200,
        'PTRATIO': students_per_classroom / 2,
        'B': 350,
        'LSTAT': 10 if poverty_level == 'high' else 5,
        'CHAS_num': 1 if next_to_river else 0,
        'CHAS_binary': 1 if next_to_river else 0,
        'CHAS_1': 1 if next_to_river else 0,
        'RAD_2': 0,
        'RAD_24': 0,
        'RAD_3': 0,
        'RAD_4': 1,
        'RAD_5': 0,
        'RAD_6': 0,
        'RAD_7': 0,
        'RAD_8': 0,
        'CHAS_label_Yes': 1 if next_to_river else 0
    }])[feature_names]

    return model.predict(input_data)[0]

# --------------------- Streamlit UI ---------------------
st.title("🏠 Boston Housing Price Predictor")
st.write("Enter house information to get an instant price prediction!")

# UI Components
next_to_river = st.checkbox("Is the house next to the river?")
nr_rooms = st.number_input("Number of rooms", min_value=1, max_value=20, value=5)
students_per_classroom = st.number_input("Students per classroom", min_value=1, max_value=100, value=20)
distance_to_town = st.number_input("Distance to town (km)", min_value=0.0, max_value=50.0, value=5.0)
pollution_level = st.selectbox("Pollution level", ["low", "high"])
poverty_level = st.selectbox("Poverty level", ["low", "high"])

# Prediction button
if st.button("Predict Price"):
    price = predict_custom_house(
        next_to_river,
        nr_rooms,
        students_per_classroom,
        distance_to_town,
        pollution_level,
        poverty_level
    )
    st.success(f" Predicted Price: {price:.2f} USD")
