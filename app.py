import streamlit as st
import pandas as pd
import pickle


# Load model and encoder

model = pickle.load(
    open("accident_model.pkl", "rb")
)

encoder = pickle.load(
    open("target_encoder.pkl", "rb")
)


# Page config

st.set_page_config(
    page_title="RoadSafe AI",
    page_icon="🚦",
    layout="wide"
)


st.title("🚦 RoadSafe AI")
st.subheader("Traffic Accident Severity Prediction")


st.write(
    "Predict accident severity using XGBoost Machine Learning"
)


st.divider()


# Input section

col1, col2 = st.columns(2)


with col1:

    age = st.selectbox(
        "Age Band of Driver",
        [
            "18-30",
            "31-50",
            "Over 51"
        ]
    )


    gender = st.selectbox(
        "Driver Gender",
        [
            "Male",
            "Female"
        ]
    )


    driving_exp = st.selectbox(
        "Driving Experience",
        [
            "Below 1yr",
            "1-5yr",
            "5-10yr",
            "Above 10yr"
        ]
    )


    vehicle = st.selectbox(
        "Vehicle Type",
        [
            "Automobile",
            "Public (> 45 seats)",
            "Lorry",
            "Motorcycle"
        ]
    )



with col2:


    weather = st.selectbox(
        "Weather Condition",
        [
            "Normal",
            "Rainy",
            "Fog"
        ]
    )


    road = st.selectbox(
        "Road Surface Condition",
        [
            "Dry",
            "Wet",
            "Snow"
        ]
    )


    light = st.selectbox(
        "Light Condition",
        [
            "Daylight",
            "Darkness - lights lit",
            "Darkness - no lighting"
        ]
    )


    cause = st.selectbox(
        "Cause of Accident",
        [
            "Speeding",
            "Driving carelessly",
            "Overtaking",
            "No distancing",
            "Other"
        ]
    )



# Prediction

if st.button("Predict Accident Severity"):


    input_data = pd.DataFrame(
        {

        "Age_band_of_driver":[age],

        "Driving_experience":[driving_exp],

        "Drivers_gender":[gender],

        "Type_of_vehicle":[vehicle],

        "Weather_conditions":[weather],

        "Road_surface_conditions":[road],

        "Light_conditions":[light],

        "Cause_of_accident":[cause]

        }
    )


    prediction = model.predict(
        input_data
    )


    result = encoder.inverse_transform(
        prediction
    )


    st.success(
        f"Predicted Severity: {result[0]}"
    )
