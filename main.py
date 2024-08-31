# from fastapi import FastAPI, HTTPException
# import pickle
# from pydantic import BaseModel
# from fastapi import FastAPI
# import uvicorn
# import joblib

# model = joblib.load('DBSCAN_model.joblib')
# scaler = joblib.load('Models/scaler.joblib')

# # model=pickle.load(open('train_model.sav','rb'))

# app = FastAPI()


# class InputFeatures(BaseModel):
#     yellow:float
#     red:float
#     position_encoded:int
    
# def preprocessing(input_features: InputFeatures):
#         dict_f = {
#     'yellow': input_features.yellow,
#         'red': input_features.red,
#     'position_encoded': input_features.position_encoded,

#     }
#         return dict_f

    
# @app.get("/predict")
# def predict(input_features: InputFeatures):
#     return preprocessing(input_features)

# # @app.get("/predict")
# # def predict(input_features: InputFeatures):
# #       dict_f = {
# #     'yellow': input_features.yellow,
# #         'red': input_features.red,
# #     'position_encoded': input_features.position_encoded,

# #     }
# #       predict=model(dict_f)

# @app.post("/predict")
# async def predict(input_features: InputFeatures):
#     data = preprocessing(input_features)
#     y_pred = model.predict(data)
#     return {"pred": y_pred.tolist()[0]}

# from fastapi import FastAPI, HTTPException
# import pickle
# from pydantic import BaseModel
# from fastapi import FastAPI
# import uvicorn
# import joblib
# import numpy as np

# # Load the trained model
# model = joblib.load('DBSCAN_model.joblib')

# # FastAPI app instance
# app = FastAPI()

# # Pydantic model for input data validation
# class InputFeatures(BaseModel):
#     yellow: float
#     red: float
#     position_encoded: int

# # Endpoint for model prediction
# @app.post("/predict")
# async def predict(data: InputFeatures):
#     features = np.array([data.yellow, data.red, data.position_encoded]).reshape(1, -1)
#     prediction = model.fit_predict(features)
#     return {"prediction": prediction.tolist()}

# Run the FastAPI app
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import streamlit as st

app = FastAPI()

# Allowing CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load('DBSCAN_model.joblib')

@app.get("/predict")
async def predict(
    yellow: float = Query(..., description="The age of the student (from 15 to 18 years)"),
    red: float = Query(..., description="The gender of the student (0: Male, 1: Female)"),
    position_encoded: int = Query(..., description="The ethnicity (0: Caucasian, 1: African American, 2: Asian, 3: Other)")
):
    predictions = model.fit_predict([[yellow, red, position_encoded]])
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

import streamlit as st
import requests

# Define the FastAPI backend URL
backend_url = "http://localhost:8000"

# Streamlit UI
st.title("Machine Learning Prediction")

yellow = st.number_input("Age of the student", min_value=15, max_value=18)
red = st.selectbox("Gender of the student", [0, 1])
position_encoded = st.selectbox("Ethnicity", [0, 1, 2, 3])

if st.button("Predict"):
    response = requests.get(f"{backend_url}/predict?yellow={yellow}&red={red}&position_encoded={position_encoded}")
    predictions = response.json()["predictions"]
    st.write("Predictions:", predictions)