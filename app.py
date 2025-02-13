import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Feature Engineering
myntra['product_avg_rating'] = myntra.groupby('name')['rating'].transform('mean')
myntra['product_rating_deviation'] = myntra['rating'] - myntra['product_avg_rating']
myntra.drop('product_avg_rating', axis=1, inplace=True)

myntra['Magic8_Product_min'] = myntra.groupby('name')['rating'].transform('min')
myntra['Magic9_Product_max'] = myntra.groupby('name')['rating'].transform('max')

# Encoding categorical features
label_encoders = {}
for col in ['sub_category', 'gender']:
    le = LabelEncoder()
    myntra[col] = le.fit_transform(myntra[col])
    label_encoders[col] = le

# Standard Scaling
scaler = StandardScaler()
scaled_features = ['product_rating_deviation', 'Magic8_Product_min', 'Magic9_Product_max']
myntra[scaled_features] = scaler.fit_transform(myntra[scaled_features])


# Load Pickle Files (For Deployment)
def load_pickle_files():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    return scaler, label_encoders

scaler, label_encoders = load_pickle_files()
