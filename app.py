import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle

csr_engine = pd.read_csv('/Users/nada/Desktop/AutoScout/Ready_to_ML.csv',index_col=None)
car_model = st.sidebar.selectbox("Choose a Car Model", ('Renault Megane', 'SEAT Leon', 'Volvo V40', 'Dacia Sandero',
                                                       'Hyundai i30', 'Opel Astra', 'Ford Mustang', 'Volvo C70',
                                                       'Peugeot 308', 'Ford Focus'))
car_type=st.sidebar.radio('Car Type', ('Used', 'Demonstration', 'Pre-registered', "Employee's car"))
age=st.sidebar.slider("Car Age (in years)", 0, 20, step=1)
mileage=st.sidebar.number_input("Car Mileage (in km)", 0, 667128, step=10)
power=st.sidebar.number_input("Car Power (kW)", 33, 450,step=1)
engine_size=st.sidebar.number_input('Engine Size ', 0, 6300, step=10)



model = pickle.load(open('/Users/nada/Desktop/AutoScout/final_model', 'rb'))


my_dict = {
"make_model": car_model,
"power_kW": power,
"mileage": mileage,
"age": age,
"engine_size": engine_size,
"type": car_type
}

df = pd.DataFrame.from_dict([my_dict])


st.header("Car Price Predictor")
st.table(df)



st.subheader("Get Price Prediction")


prediction = model.predict(df)

if st.button("Predict Price"):
 st.success("The estimated price is {} ".format(int(prediction)))

st.write("\n\n")



