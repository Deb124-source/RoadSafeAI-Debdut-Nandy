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


    # create dataframe with ALL training columns

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=features
    )


    # fill user selected values

    if "Age_band_of_driver" in features:
        input_data["Age_band_of_driver"] = age


    if "Drivers_gender" in features:
        input_data["Drivers_gender"] = gender


    if "Driving_experience" in features:
        input_data["Driving_experience"] = experience


    if "Type_of_vehicle" in features:
        input_data["Type_of_vehicle"] = vehicle


    if "Weather_conditions" in features:
        input_data["Weather_conditions"] = weather


    if "Road_surface_conditions" in features:
        input_data["Road_surface_conditions"] = road


    if "Light_conditions" in features:
        input_data["Light_conditions"] = light


    if "Cause_of_accident" in features:
        input_data["Cause_of_accident"] = cause



    # Prediction

    prediction = model.predict(
        input_data
    )


    result = encoder.inverse_transform(
        prediction
    )


    st.success(
        f"Predicted Accident Severity: {result[0]}"
    )
