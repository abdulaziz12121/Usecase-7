import streamlit as st
import requests

# Set the URL for the FastAPI app
url_dbscan = "https://usecase-7-12ia.onrender.com/predict_dbscan"  # Update with your actual endpoint

# Create the Streamlit app
st.title("Player Value and Performance DBSCAN")

# Input fields for the user to provide data
yellow = st.number_input("yellow", min_value=0.0, max_value=1.0, value=0.1)

red = st.number_input("red", min_value=0.0, max_value=1.0, value=0.1)

position_encoded = st.number_input("position_encoded", min_value=0, max_value=4, value=0, step=1)
data = {
    "yellow": yellow,
    "red": red,
    "position_encoded": position_encoded,
}

# Button to trigger the KMeans prediction

# Button to trigger the DBSCAN prediction
if st.button("Predict DBSCAN"):
    # Send the data to FastAPI and get the response
    response = requests.post(url_dbscan, json=data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"The predicted DBSCAN  is: {result['dbscan_pred']}")
    else:
        st.error("Failed to get a DBSCAN prediction. Please try again.")
