import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

# Load Pickle Files (For Deployment)
def load_pickle_files():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return scaler, label_encoders, model

# Load model and preprocessing tools
scaler, label_encoders, model = load_pickle_files()

# App title
st.title("Car Price Prediction App")

# Input fields
sub_category = st.selectbox("Select Sub Category", options=list(label_encoders['sub_category'].classes_))
gender = st.selectbox("Select Gender", options=list(label_encoders['gender'].classes_))
product_rating_deviation = st.number_input("Product Rating Deviation", value=0.0)
magic8_product_min = st.number_input("Magic8 Product Min", value=0.0)
magic9_product_max = st.number_input("Magic9 Product Max", value=0.0)

# Encoding inputs
sub_category_encoded = label_encoders['sub_category'].transform([sub_category])[0]
gender_encoded = label_encoders['gender'].transform([gender])[0]

# Scale numerical inputs
scaled_features = scaler.transform([[product_rating_deviation, magic8_product_min, magic9_product_max]])

# Predict button
if st.button("Predict Selling Price"):
    features = np.array([[sub_category_encoded, gender_encoded, scaled_features[0][0], scaled_features[0][1], scaled_features[0][2]]])
    prediction = model.predict(features)
    st.success(f"Predicted Selling Price: ${prediction[0]:,.2f}")
