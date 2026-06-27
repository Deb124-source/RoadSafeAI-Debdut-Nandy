import streamlit as st
import pandas as pd
import joblib


# Load saved files

model = joblib.load(
    "acc_model.pkl"
)

encoder = joblib.load(
    "model_encoder.pkl"
)

target_encoder = joblib.load(
    "targeting_encoder.pkl"
)

features = joblib.load(
    "model_features.pkl"
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


# Input fields

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


    # Create raw input dataframe

    input_data = pd.DataFrame(
        {

        "Age_band_of_driver":[age],

        "Drivers_gender":[gender],

        "Driving_experience":[experience],

        "Type_of_vehicle":[vehicle],

        "Weather_conditions":[weather],

        "Road_surface_conditions":[road],

        "Light_conditions":[light],

        "Cause_of_accident":[cause]

        }
    )



    # Encode categorical values

    encoded_data = encoder.transform(
        input_data
    )


    encoded_data = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out()
    )



    # Combine with numeric defaults

    final_input = encoded_data.copy()



    # Match training columns exactly

    final_input = final_input.reindex(
        columns=features,
        fill_value=0
    )



    # Prediction

    prediction = model.predict(
        final_input
    )


    result = target_encoder.inverse_transform(
        prediction
    )



    st.success(
        f"Predicted Accident Severity: {result[0]}"
    )
