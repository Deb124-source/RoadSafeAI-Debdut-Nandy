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

encoder_features = encoder.feature_names_in_



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


    # Create dataframe with ALL encoder columns

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=encoder.feature_names_in_
    )


    # Fill available user inputs

    input_data["Age_band_of_driver"] = age

    input_data["Drivers_gender"] = gender

    input_data["Driving_experience"] = experience

    input_data["Type_of_vehicle"] = vehicle

    input_data["Weather_conditions"] = weather

    input_data["Road_surface_conditions"] = road

    input_data["Light_conditions"] = light

    input_data["Cause_of_accident"] = cause



    # Give default categorical values for remaining columns

    defaults = {

        "Day_of_week":"Monday",

        "Educational_level":"Unknown",

        "Vehicle_driver_relation":"Owner",

        "Owner_of_vehicle":"Owner",

        "Service_year_of_vehicle":"Unknown",

        "Defect_of_vehicle":"No defect",

        "Area_accident_occured":"Other",

        "Lanes_or_Medians":"Two-way",

        "Road_allignment":"Straight",

        "Types_of_Junction":"No junction",

        "Road_surface_type":"Asphalt",

        "Type_of_collision":"Other",

        "Vehicle_movement":"Going straight",

        "Casualty_class":"Driver",

        "Casualty_gender":"Male",

        "Age_band_of_casualty":"18-30",

        "Work_of_casuality":"Unknown",

        "Fitness_of_casuality":"Normal",

        "Pedestrian_movement":"Not pedestrian"

    }



    for col,value in defaults.items():

        if col in input_data.columns:

            input_data[col] = value



    # Encode

    encoded_data = encoder.transform(
        input_data
    )


    encoded_data = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(
            encoder.feature_names_in_
        )
    )


    # Match XGBoost input

    final_input = encoded_data.reindex(
        columns=features,
        fill_value=0
    )


    prediction = model.predict(
        final_input
    )


    result = target_encoder.inverse_transform(
        prediction
    )


    st.success(
        f"Predicted Accident Severity: {result[0]}"
    )
