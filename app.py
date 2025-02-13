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
        label_encoders = pickle.load(f)  # Now correctly stored as a dictionary
    
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return scaler, label_encoders, model

# Load model and preprocessing tools
scaler, label_encoders, model = load_pickle_files()

# Retrieve LabelEncoders from dictionary
sub_category_encoder = label_encoders.get('sub_category')
gender_encoder = label_encoders.get('gender')

if sub_category_encoder and gender_encoder:
    sub_category = st.selectbox("Select Sub Category", options=list(sub_category_encoder.classes_))
    gender = st.selectbox("Select Gender", options=list(gender_encoder.classes_))

    # Encoding inputs
    sub_category_encoded = sub_category_encoder.transform([sub_category])[0]
    gender_encoded = gender_encoder.transform([gender])[0]
else:
    st.error("Encoding files not loaded properly. Please check label_encoder.pkl.")
    sub_category_encoded, gender_encoded = None, None

# Input fields for 15 features
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Product Rating Deviation", value=0.0)
feature_4 = st.number_input("Magic8 Product Min", value=0.0)
feature_5 = st.number_input("Magic9 Product Max", value=0.0)
feature_6 = st.number_input("Feature 6", value=0.0)
feature_7 = st.number_input("Feature 7", value=0.0)
feature_8 = st.number_input("Feature 8", value=0.0)
feature_9 = st.number_input("Feature 9", value=0.0)
feature_10 = st.number_input("Feature 10", value=0.0)
feature_11 = st.number_input("Feature 11", value=0.0)
feature_12 = st.number_input("Feature 12", value=0.0)
feature_13 = st.number_input("Feature 13", value=0.0)
feature_14 = st.number_input("Feature 14", value=0.0)
feature_15 = st.number_input("Feature 15", value=0.0)

# Create feature array
input_features = np.array([
    [
        feature_1, feature_2, feature_3, feature_4, feature_5,
        feature_6, feature_7, feature_8, feature_9, feature_10,
        feature_11, feature_12, feature_13, feature_14, feature_15
    ]
])

# Transform correctly with 15 features
scaled_features = scaler.transform(input_features)

# Predict button
if st.button("Predict Selling Price") and sub_category_encoded is not None and gender_encoded is not None:
    features = np.array([
        [
            sub_category_encoded, gender_encoded,
            scaled_features[0][0], scaled_features[0][1], scaled_features[0][2],
            scaled_features[0][3], scaled_features[0][4], scaled_features[0][5],
            scaled_features[0][6], scaled_features[0][7], scaled_features[0][8],
            scaled_features[0][9], scaled_features[0][10], scaled_features[0][11],
            scaled_features[0][12], scaled_features[0][13], scaled_features[0][14]
        ]
    ])
    prediction = model.predict(features)
    st.success(f"Predicted Rating: ${prediction[0]:,.2f}")
