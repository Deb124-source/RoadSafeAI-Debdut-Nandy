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



# Page settings

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



# Input Layout

col1, col2 = st.columns(2)



with col1:


    day = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )


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


    education = st.selectbox(
        "Educational Level",
        [
            "High school",
            "Junior high school",
            "Above high school",
            "Unknown"
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


    owner = st.selectbox(
        "Owner of Vehicle",
        [
            "Owner",
            "Organization"
        ]
    )


    cause = st.selectbox(
    "Cause of Accident",
    [
        "Driving carelessly",
        "Speeding",
        "Overtaking",
        "No distancing",
        "Other"
    ]
)
with col2:


    area = st.selectbox(
        "Accident Area",
        [
            "Residential areas",
            "Office areas",
            "Industrial areas",
            "Market areas"
        ]
    )


    road_alignment = st.selectbox(
        "Road Alignment",
        [
            "Tangent road",
            "Steep grade downward",
            "Curve road"
        ]
    )


    junction = st.selectbox(
        "Type of Junction",
        [
            "No junction",
            "Y Shape",
            "Crossing",
            "O Shape"
        ]
    )


    road_surface = st.selectbox(
        "Road Surface Type",
        [
            "Asphalt roads",
            "Earth roads",
            "Gravel roads"
        ]
    )


    road_condition = st.selectbox(
        "Road Surface Condition",
        [
            "Dry",
            "Wet or damp",
            "Snow"
        ]
    )


    weather = st.selectbox(
        "Weather Condition",
        [
            "Normal",
            "Raining",
            "Fog"
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


    collision = st.selectbox(
        "Type of Collision",
        [
            "Vehicle with vehicle collision",
            "Collision with roadside object",
            "Collision with pedestrian"
        ]
    )


    movement = st.selectbox(
        "Vehicle Movement",
        [
            "Going straight",
            "Overtaking",
            "Changing lane"
        ]
    )


st.divider()



if st.button("Predict Severity"):


    # Create dataframe matching encoder

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=encoder_features
    )


    values = {


        "Day_of_week": day,

        "Age_band_of_driver": age,

        "Drivers_gender": gender,

        "Educational_level": education,

        "Driving_experience": experience,

        "Type_of_vehicle": vehicle,

        "Owner_of_vehicle": owner,

        "Area_accident_occured": area,

        "Road_allignment": road_alignment,

        "Types_of_Junction": junction,

        "Road_surface_type": road_surface,

        "Road_surface_conditions": road_condition,

        "Weather_conditions": weather,

        "Light_conditions": light,

        "Type_of_collision": collision,

        "Vehicle_movement": movement,


        # defaults

        "Vehicle_driver_relation":"Owner",

        "Service_year_of_vehicle":"Unknown",

        "Defect_of_vehicle":"No defect",

        "Lanes_or_Medians":"Two-way",

        "Casualty_class":"Driver",

        "Casualty_gender":"Male",

        "Age_band_of_casualty":"18-30",

        "Work_of_casuality":"Unknown",

        "Fitness_of_casuality":"Normal",

        "Pedestrian_movement":"Not pedestrian",

        "Cause_of_accident":"Driving carelessly"

    }



    for col,value in values.items():

        if col in input_data.columns:

            input_data[col] = value



    # Encode

    encoded = encoder.transform(
        input_data
    )


    encoded = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out()
    )


    # Match XGBoost columns

    final_input = encoded.reindex(
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
        f"🚦 Predicted Accident Severity: {result[0]}"
    )



    # Probability

    if hasattr(model,"predict_proba"):

        probability = model.predict_proba(
            final_input
        )[0]


        st.subheader("Prediction Confidence")


        prob_df = pd.DataFrame(
            {
                "Severity":
                target_encoder.classes_,

                "Probability":
                probability*100
            }
        )


        st.dataframe(
            prob_df.style.format(
                {
                    "Probability":"{:.2f}%"
                }
            )
        )
