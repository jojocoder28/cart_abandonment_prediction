import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from sklearn.pipeline import Pipeline

st.title('Chop The Data - Cart Abandonment Prediction')

df = pd.read_csv("IIMK_Coherence6_Case_Dataset.csv")
# Load the trained model and label encoders
model = joblib.load("cart_abandonment_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Configure GenAI with Gemini API
genai.configure(api_key="AIzaSyAa6nFL9873prf7EOUdoySjL41pV_K2vNs")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Define the input features and corresponding options
features = [
    'Gender', 'Age', 'City', 'Product_Category', 'Price', 'Quantity',
    'Payment_Method', 'Browsing_Time_mins', 
    'Discount_Applied', 'Loyalty_Points_Earned'
]
st.title("Cart Abandonment Prediction App")
st.write("Predict cart abandonment and get actionable insights to reduce it.")
# Define options for categorical features
user_inputs = {}

# Categorical inputs
user_inputs['Gender'] = st.selectbox('Gender', options=df['Gender'].unique())
user_inputs['City'] = st.selectbox('City', options=df['City'].unique())
user_inputs['Product_Category'] = st.selectbox('Product Category', options=df['Product_Category'].unique())
user_inputs['Payment_Method'] = st.selectbox('Payment Method', options=df['Payment_Method'].unique())

# Numerical inputs
user_inputs['Age'] = st.number_input('Age', min_value=0, step=1)
user_inputs['Price'] = st.number_input('Price', min_value=0.0, step=0.1)
user_inputs['Quantity'] = st.number_input('Quantity', min_value=1, step=1)
user_inputs['Browsing_Time_mins'] = st.number_input('Browsing Time (in minutes)', min_value=0.0, step=0.1)
user_inputs['Discount_Applied'] = st.number_input('Discount Applied (%)', min_value=0.0, step=0.1)
user_inputs['Loyalty_Points_Earned'] = st.number_input('Loyalty Points Earned', min_value=0, step=1)

# Predict button
if st.button("Predict Cart Abandonment"):
    # Encode categorical inputs
    for feature in ['Gender', 'City', 'Product_Category', 'Payment_Method']:
        user_inputs[feature] = label_encoders[feature].transform([user_inputs[feature]])[0]
    
    # Convert inputs to DataFrame using predefined feature order
    input_df = pd.DataFrame([[user_inputs[feature] for feature in features]], columns=features)

    # Predict cart abandonment
    prediction = model.predict(input_df)[0]
    abandonment_flag = "Yes" if prediction == 1 else "No"
    
    st.subheader("Prediction")
    st.write(f"Cart Abandonment Flag: {abandonment_flag}")

    # Generate feedback using Gemini API
    feedback_prompt = (
        f"The user data is: {user_inputs}. The cart abandonment flag is '{abandonment_flag}'. "
        f"Provide actionable insights to reduce cart abandonment rates and improve user experience."
    )
    feedback_response = gemini_model.generate_content(feedback_prompt)
    feedback = feedback_response.text

    st.subheader("AI enabled Feedback & Recommendation")
    st.write(feedback)
