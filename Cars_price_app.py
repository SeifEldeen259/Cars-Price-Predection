import streamlit as st
import joblib
import numpy as np
import os

# Sidebar
st.sidebar.title("Car Price Prediction")
st.sidebar.image("https://github.com/SeifEldeen259/Cars-Price-Predection/blob/main/Cars.jpg?raw=true", use_container_width=True)
st.sidebar.markdown("### Enter Car Details:")
year = st.sidebar.number_input("Year of Manufacture", min_value=1900, max_value=2025, step=1, value=2015)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=1, value=50000)
mileage = st.sidebar.number_input("Mileage (kmpl)", min_value=0.0, step=0.1, value=18.0)
fuel = st.sidebar.selectbox("Fuel Type", ["Diesel", "Petrol", "LPG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Owner Type", ["First Owner", "Second Owner", "Third and Above"])  # Combined "Third" & "Fourth"
engine = st.sidebar.number_input("Engine (CC)", min_value=0.0, step=0.1, value=1197.0)
max_power = st.sidebar.number_input("Max Power (BHP)", min_value=0.0, step=0.1, value=82.0)
seats = st.sidebar.number_input("Seats", min_value=1, max_value=10, step=1, value=5)

# Preprocess inputs
car_age = 2025 - year
categorical_mapping = {
    "Fuel Type": {"Diesel": 0, "Petrol": 1, "LPG": 2},
    "Seller Type": {"Individual": 0, "Dealer": 1, "Trustmark Dealer": 2},
    "Transmission": {"Manual": 0, "Automatic": 1},
    "Owner Type": {"First Owner": 0, "Second Owner": 1, "Third and Above": 2}  # Updated to "Third and Above"
}

fuel = categorical_mapping["Fuel Type"][fuel]
seller_type = categorical_mapping["Seller Type"][seller_type]
transmission = categorical_mapping["Transmission"][transmission]
owner = categorical_mapping["Owner Type"][owner]

# Prepare input array with the exact number of features used during training
input_features = np.array([[
    km_driven, mileage, engine, max_power, seats, car_age, 
    fuel == 0, fuel == 1, fuel == 2,  # One-hot encode fuel
    seller_type == 0, seller_type == 1, seller_type == 2,  # One-hot encode seller_type
    transmission == 0,  # One-hot encode transmission
    owner == 0, owner == 1, owner == 2  # One-hot encode owner (First, Second, Third and Above)
]])

# Main Page
st.title("Car Price Prediction App")
st.image("https://github.com/SeifEldeen259/Cars-Price-Predection/blob/main/bugatti.jpg?raw=true", use_container_width=True)
st.markdown("""
### About the Model
This car price prediction tool leverages a sophisticated machine learning model trained on extensive car data. 
It uses Gradient Boosting, a powerful algorithm known for its accuracy in regression tasks. 
The model considers multiple factors such as car age, mileage, engine power, and ownership history 
to deliver reliable price estimates. Its goal is to help users make informed decisions when buying or selling cars.
""")

st.image("https://github.com/SeifEldeen259/Cars-Price-Predection/blob/main/Car.jpg?raw=true", use_container_width=True)

st.markdown("""
### Information Needed
To predict the price, please provide:
- The year the car was manufactured.
- The distance driven in kilometers.
- Fuel type, transmission, and ownership details.
- Additional specifications like mileage, engine power, and seating capacity.
""")

try:
    # Dynamically locate the model
    model_path = os.path.join(os.path.dirname(__file__), 'car_model.pkl')
    model = joblib.load(model_path)
    st.write("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Ensure 'car_model.pkl' is present in the same directory.")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.write("Deployment environment details:", os.listdir(os.path.dirname(__file__)))


# Prediction Result Display
if st.sidebar.button("Predict"):
    predicted_price = model.predict(input_features)[0]
    st.markdown(f"### Estimated Price: ₹{predicted_price:,.2f}")
