import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

HF_USERNAME = "Bhavesh-Hug23"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="best_model.joblib", repo_type="model")
    columns_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="feature_columns.joblib", repo_type="model")
    return joblib.load(model_path), joblib.load(columns_path)

model, feature_columns = load_model()

st.title("Wellness Tourism Package - Purchase Prediction")
st.write("Enter customer details to predict whether they are likely to purchase the Wellness Tourism Package.")

age = st.number_input("Age", 18, 100, 35)
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 0, 60, 10)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_persons = st.number_input("Number of Persons Visiting", 1, 10, 2)
num_followups = st.number_input("Number of Followups", 0, 10, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips per Year", 0, 20, 2)
passport = st.selectbox("Passport", [0, 1])
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Own Car", [0, 1])
num_children = st.number_input("Number of Children Visiting", 0, 5, 0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", 0, value=20000)

if st.button("Predict"):
    input_dict = {
        "Age": age, "CityTier": city_tier, "DurationOfPitch": duration_of_pitch,
        "NumberOfPersonVisiting": num_persons, "NumberOfFollowups": num_followups,
        "PreferredPropertyStar": preferred_star, "NumberOfTrips": num_trips,
        "Passport": passport, "PitchSatisfactionScore": pitch_score, "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children, "MonthlyIncome": monthly_income,
        "TypeofContact": typeof_contact, "Occupation": occupation, "Gender": gender,
        "ProductPitched": product_pitched, "MaritalStatus": marital_status,
        "Designation": designation,
    }
    input_df = pd.get_dummies(pd.DataFrame([input_dict]))
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Package (probability: {probability:.2%})")
    else:
        st.warning(f"Unlikely to purchase the Wellness Package (probability: {probability:.2%})")
