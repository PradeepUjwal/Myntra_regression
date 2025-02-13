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

# Retrieve LabelEncoders
sub_category_encoder = label_encoders.get('sub_category')
gender_encoder = label_encoders.get('gender')

if sub_category_encoder and gender_encoder:
    sub_category = st.selectbox("Select Sub Category", options=list(sub_category_encoder.classes_))
    gender = st.selectbox("Select Gender", options=list(gender_encoder.classes_))

    # Encoding inputs
    sub_category_encoded = sub_category_encoder.transform([sub_category])[0]
    gender_encoded = gender_encoder.transform([gender])[0]
else:
    st.error("Encoding files not loaded properly. Please check label_encoders.pkl.")
    sub_category_encoded, gender_encoded = None, None

# Input fields
product_rating_deviation = st.number_input("Product Rating Deviation", value=0.0)
magic8_product_min = st.number_input("Magic8 Product Min", value=0.0)
magic9_product_max = st.number_input("Magic9 Product Max", value=0.0)

# Scale numerical inputs
scaled_features = scaler.transform([[product_rating_deviation, magic8_product_min, magic9_product_max]])

# Predict button
if st.button("Predict Selling Price") and sub_category_encoded is not None and gender_encoded is not None:
    features = np.array([[sub_category_encoded, gender_encoded, scaled_features[0][0], scaled_features[0][1], scaled_features[0][2]]])
    prediction = model.predict(features)
    st.success(f"Predicted Selling Price: ${prediction[0]:,.2f}")
