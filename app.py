import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# App title
st.title("Student Performance Predictor")

st.write("Enter student details to predict the final grade.")
st.header("Student Information")

school = st.selectbox("School", ["GP", "MS"])

sex = st.selectbox("Gender", ["F", "M"])

age = st.number_input("Age", min_value=15, max_value=22, value=17)

address = st.selectbox("Address", ["U", "R"])

famsize = st.selectbox("Family Size", ["LE3", "GT3"])
Pstatus = st.selectbox("Parent Status", ["A", "T"])

Medu = st.slider("Mother's Education", 0, 4, 2)

Fedu = st.slider("Father's Education", 0, 4, 2)

Mjob = st.selectbox(
    "Mother's Job",
    ["teacher", "health", "services", "at_home", "other"]
)

Fjob = st.selectbox(
    "Father's Job",
    ["teacher", "health", "services", "at_home", "other"]
)
reason = st.selectbox(
    "Reason for choosing school",
    ["home", "reputation", "course", "other"]
)

guardian = st.selectbox(
    "Guardian",
    ["mother", "father", "other"]
)

traveltime = st.slider(
    "Travel Time",
    1, 4, 1
)

studytime = st.slider(
    "Study Time",
    1, 4, 2
)

failures = st.slider(
    "Number of Previous Failures",
    0, 3, 0
)
schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])

famsup = st.selectbox("Family Educational Support", ["yes", "no"])

paid = st.selectbox("Extra Paid Classes", ["yes", "no"])

activities = st.selectbox("Extra-Curricular Activities", ["yes", "no"])

nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
higher = st.selectbox("Wants Higher Education", ["yes", "no"])

internet = st.selectbox("Internet Access at Home", ["yes", "no"])

romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])

famrel = st.slider("Family Relationship Quality", 1, 5, 4)

freetime = st.slider("Free Time After School", 1, 5, 3)
goout = st.slider("Going Out with Friends", 1, 5, 3)

Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)

Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)

health = st.slider("Current Health Status", 1, 5, 3)

absences = st.number_input(
    "Number of School Absences",
    min_value=0,
    max_value=93,
    value=5
)
if st.button("Predict Final Grade"):

    # Create input dataframe
    input_data = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences
    }])

    # Preprocess input
    input_encoded = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_encoded)[0]

    # Display prediction
    st.success(f"Predicted Final Grade: {prediction:.2f} / 20")