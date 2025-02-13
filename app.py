import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Data (Replace with actual data loading logic)
data = pd.read_csv('myntra1.csv')

# Feature Engineering
data['product_avg_rating'] = data.groupby('name')['rating'].transform('mean')
data['product_rating_deviation'] = data['rating'] - data['product_avg_rating']
data.drop('product_avg_rating', axis=1, inplace=True)

data['Magic8_Product_min'] = data.groupby('name')['rating'].transform('min')
data['Magic9_Product_max'] = data.groupby('name')['rating'].transform('max')

# Encoding categorical features
label_encoders = {}
for col in ['sub_category', 'gender']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Standard Scaling
scaler = StandardScaler()
scaled_features = ['product_rating_deviation', 'Magic8_Product_min', 'Magic9_Product_max']
data[scaled_features] = scaler.fit_transform(data[scaled_features])


# Save Processed Data (for further model training)
data.to_csv('processed_myntra.csv', index=False)

# Load Pickle Files (For Deployment)
def load_pickle_files():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    return scaler, label_encoders

scaler, label_encoders = load_pickle_files()
