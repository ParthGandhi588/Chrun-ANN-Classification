import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model("D:\\Udemy Course\\NLP & Deep Learning\\ANN Classification\\models\\model.h5")

# Load the scaler pickle
scaler = pickle.load(open("D:\\Udemy Course\\NLP & Deep Learning\\ANN Classification\\models\\scaler.pkl", 'rb'))

# Load the label encoder pickle
label_encoder_gender = pickle.load(open("D:\\Udemy Course\\NLP & Deep Learning\\ANN Classification\\models\\label_encoder_gender.pkl", 'rb'))

# Load the onehot encoder pickle
onehot_encoder_geo = pickle.load(open("D:\\Udemy Course\\NLP & Deep Learning\\ANN Classification\\models\\onehot_encoder_geo.pkl", 'rb'))

# Stramlit app
st.title("Customer Churn Prediction")

#  Input fields
Geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
Gender = st.selectbox("Gender", label_encoder_gender.classes_)
Age = st.slider("Age", 18,92)
Balance = st.number_input("Balance")
credit_score = st.number_input("CreditScore")
EstimatedSalary = st.number_input("EstimatedSalary", 0)
Tenure = st.slider("Tenure", 0,10)
NumOfProducts = st.slider("NumOfProducts", 1,4)
HasCrCard = st.selectbox("HasCrCard", [0,1])
IsActiveMember = st.selectbox("IsActiveMember", [0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine the one-hot encoded columns with other features
final_input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data (excluding 'Geography' column as it is already encoded)
input_data_scaled = scaler.transform(final_input_data)

# Make predictions
predictions = model.predict(input_data_scaled)
prediction_prob = predictions[0][0]

st.write(f'Prediction Probability: {prediction_prob: .2f}')

if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")
