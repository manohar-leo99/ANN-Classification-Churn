import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle


# Load the trained model

model = tf.keras.models.load_model('model.h5')


# The Encoder and scaler used during training

with open('lable_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehotencoder.pkl', 'rb') as file:
    onehotencoder = pickle.load(file)



# Streamlit App
st.title("Bank Customer Churn Prediction")

# Input fields
geography = st.selectbox('Geography', onehotencoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance')
Credit_Score = st.number_input('Credit Score')
Estemated_Salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
num_of_products = st.number_input('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'Geography': [geography],
    'Gender':   [label_encoder_gender.transform([gender])[0]],
    'Age': [age], 
    'Balance': [balance],
    # Column names must match training-time feature names used by the scaler
    'CreditScore': [Credit_Score],
    'EstimatedSalary': [Estemated_Salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# One-hot encode 'Geography'
geo_encoded = onehotencoder.transform([[geography]]).toarray()
geo_encode_df = pd.DataFrame(geo_encoded, columns=onehotencoder.get_feature_names_out(['Geography']))

# Concatenate the one-hot encoded geography columns with the input dataframe
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encode_df], axis=1)

# Reorder columns to match the scaler's training feature order
required_columns = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]
input_data = input_data.reindex(columns=required_columns)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction churn

prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

if st.button('Predict Churn'):
    if prediction_probability > 0.5:
        st.write(f"The customer is likely to leave the bank with a probability of {prediction_probability:.2f}")
    else:
        st.write(f"The customer is likely to stay with the bank with a probability of {1 - prediction_probability:.2f}")