# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

#Load dataset
df = pd.read_csv("output/amazon_model_dataset.csv")
clean_df = pd.read_csv("output/amazon_cleaned_data.csv")

#Create title
st.title("Amazon Delivery Performance Dashboard")

#Create our KPI Section 
col1, col2 = st.columns([1,1])

with col1:
    total = len(df)
    st.metric("Total Deliveries", total)
with col2: 
    slow_rate = df["Slow_Delivery"].mean() * 100
    st.metric("Slow Delivery Rate (%)", round(slow_rate, 2))

#EDA SECTION
st.header("Exploratory Data Analysis")

#create 2 columns for 2 charts
col_chart1, col_chart2 = st.columns([2,2])

#First bar chart 
with col_chart1:
    area_data = clean_df.groupby("Area")["Delivery_Time"].median()
    
    fig, ax = plt.subplots()
    area_data.plot(kind ="bar", ax=ax)
    
    ax.set_ylabel("Median Delivery Time")
    ax.set_xlabel("Area")
    ax.set_title("Median Delivery Time by Area")
    st.pyplot(fig)

with col_chart2:
    weather_data = clean_df.groupby("Weather")["Delivery_Time"].median()

    fig, ax = plt.subplots()
    weather_data.plot(kind="bar", ax=ax)
    
    ax.set_ylabel("Median Delivery Time")
    ax.set_xlabel("Weather")
    ax.set_title("Median Delivery Time by Weather Type")
    st.pyplot(fig)

#Model Training Section

st.header("Predict Delivery Delay Risk")

# Prepare data
X = df.drop("Slow_Delivery", axis=1)
y = df["Slow_Delivery"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

#User Input Section

st.subheader("Enter Delivery Details")

#create 4 columns for 4 selection boxes

col_box1, col_box2, col_box3, col_box4 = ([3,3,3,3])

with col_box1: 
    distance = st.number_input("Distance (km)", min_value=0.0, value=10.0)
with col_box2: 
    weather = st.selectbox("Weather", [col.replace("Weather_", "") for col in X.columns if "Weather_" in col])
with col_box3: 
    area = st.selectbox("Area", [col.replace("Area_", "") for col in X.columns if "Area_" in col])
with col_box4: 
    category = st.selectbox("Category", [col.replace("Category_", "") for col in X.columns if "Category_" in col])

def make_prediction(distance, weather, area, category):
    row = pd.DataFrame([np.zeros(len(X.columns))], columns=X.columns)

    row["Distance_in_km"] = distance

    if f"Weather_{weather}" in row.columns:
        row[f"Weather_{weather}"] = 1
    if f"Area_{area}" in row.columns:
        row[f"Area_{area}"] = 1
    if f"Category_{category}" in row.columns:
        row[f"Category_{category}"] = 1

    return model.predict(row)[0]
# Create Predict Button
if st.button("Predict Delay Risk"):
    result = make_prediction(distance, weather, area, category)

    if result == 1:
        st.error("High Risk: Slow Delivery Expected")
    else:
        st.success("Low Risk: On-Time Delivery Expected")
