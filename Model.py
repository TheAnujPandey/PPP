# Model.py

import subprocess
import sys

# Install joblib if not installed
try:
    import joblib
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib

# Rest of the imports
import numpy as np
import streamlit as st

# Load the model
model = joblib.load('Housing_Model.joblib')# features = np.array([0.36234185, -0.49612729, 0.1201202, -0.07890437, -1.39536133, -0.4020606, 1.06076002, 0.0])
# features = features.reshape(1, -1)  

# print ("predicted values", model.predict)

st.title("Housing value Prediction")


import streamlit as st

col1, col2, = st.columns(2)

No = col1.number_input(label="No", min_value=0)
Date = col2.number_input(label="X1 transaction date", min_value=0)
Age = col2.number_input("X2 house age", min_value=0)
Distance = col1.number_input("X3 distance to the nearest MRT station", min_value=0)
stores = col2.number_input("X4 number of convenience stores", min_value=0)
Latitude = col1.number_input("X5 latitude", min_value=0)
Longitude = col2.number_input("X6 longitude", min_value=0)
Sq = col2.number_input("Y house price of unit area", min_value=0)



button = st.button("Predict")

if button:
    try:
        # Debugging: Print input features
        print("Input Features:", No, Date, Age, Distance, stores, Latitude, Longitude, Sq)

        # Prepare the input data
        features = np.array([No, Date, Age, Distance, stores, Latitude, Longitude, Sq])
        features = features.reshape(1, -1)

        # Debugging: Print loaded model's parameters
        print("Model Parameters:", model.get_params())

        # Debugging: Print feature importance if applicable
        if hasattr(model, 'feature_importances_'):
            print("Feature Importances:", model.feature_importances_)

        # Make a prediction using the loaded model
        prediction = model.predict(features)

        # Debugging: Print prediction
        print("Model Prediction:", prediction)

        # Display the prediction
        st.success(f"The predicted house price of unit area is {prediction[0]:.2f} units.")
    except Exception as e:
        st.error(f"Error: {str(e)}")


# if button:
#     try:
#         # Debugging: Print input features
#         print("Input Features:", No, Date, Age, Distance, stores, Latitude, Longitude, Sq)

#         # Prepare the input data
#         features = np.array([No, Date, Age, Distance, stores, Latitude, Longitude, Sq])
#         features = features.reshape(1, -1)

#         # Debugging: Print loaded model's parameters
#         print("Model Parameters:", model.get_params())

#         # Make a prediction using the loaded model
#         prediction = model.predict(features)

#         # Debugging: Print prediction
#         print("Model Prediction:", prediction)

#         # Display the prediction
#         st.success(f"The predicted house price of unit area is {prediction[0]:.2f} units.")
#     except Exception as e:
#         st.error(f"Error: {str(e)}")
# if button:
#     try:
#         # Prepare the input data
#         features = np.array([No, Date, Age, Distance, stores, Latitude, Longitude, Sq])
#         features = features.reshape(1, -1)

#         # Make a prediction using the loaded model
#         prediction = model.predict(features)

#         # Display the prediction
#         st.success(f"The predicted house price of unit area is {prediction[0]:.2f} units.")
#     except Exception as e:
#         st.error(f"Error: {str(e)}")





