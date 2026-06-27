import streamlit as st
import pandas as pd
import pickle


# Load files

model = pickle.load(
    open("accident_model (1).pkl", "rb")
)

encoder = pickle.load(
    open("target_encoder (1).pkl", "rb")
)

features = pickle.load(
    open("features (1).pkl", "rb")
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
    "Predict accident severity using XGBoost"
)


st.divider()



# Inputs

col1, col2 = st.columns(2)


with col1:

    age = st.selectbox(
        "Age Band of Driver",
        [
            "Under 18",
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


    experience = st.selectbox(
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
            "Lorry",
            "Motorcycle",
            "Public"
        ]
    )



with col2:


    weather = st.selectbox(
        "Weather Condition",
        [
            "Normal",
            "Raining",
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
            "Darkness"
        ]
    )


    cause = st.selectbox(
        "Cause of Accident",
        [
            "Speeding",
            "Overtaking",
            "Driving carelessly",
            "Other"
        ]
    )



st.divider()



if st.button("Predict Severity"):

    # create dataframe with correct columns
    input_data = pd.DataFrame(
        columns=features
    )


    # add one row
    input_data = pd.DataFrame(
        [[0]*len(features)],
        columns=features
    )


    # Fill categorical values

    categorical_values = {

        "Age_band_of_driver": age,

        "Drivers_gender": gender,

        "Driving_experience": experience,

        "Type_of_vehicle": vehicle,

        "Weather_conditions": weather,

        "Road_surface_conditions": road,

        "Light_conditions": light,

        "Cause_of_accident": cause

    }


    for col, val in categorical_values.items():

        if col in input_data.columns:
            input_data[col] = str(val)



    # Fill numeric columns

    numeric_values = {

        "Number_of_vehicles_involved": 1,

        "Number_of_casualties": 1

    }


    for col, val in numeric_values.items():

        if col in input_data.columns:
            input_data[col] = val



    # Ensure numeric columns are numeric

    for col in input_data.columns:

        if col not in categorical_values:

            input_data[col] = pd.to_numeric(
                input_data[col],
                errors="coerce"
            )



    prediction = model.predict(
        input_data
    )


    result = encoder.inverse_transform(
        prediction
    )


    st.success(
        f"Predicted Accident Severity: {result[0]}"
    )
