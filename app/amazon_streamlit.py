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


tab1, tab2, tab3 = st.tabs(["Overview", "EDA", "Prediction"])

#Create title
st.title("Amazon Delivery Performance Dashboard")

#Create our KPI Section 
col1, col2, col3 = st.columns([1,1,1])

with col1:
    total = len(df)
    st.metric("Total Deliveries", total)
with col2: 
    slow_rate = df["Slow_Delivery"].mean() * 100
    st.metric("Slow Delivery Rate (%)", round(slow_rate, 2))
with col3: 
    avg_time = clean_df["Delivery_Time"].mean()
    st.metric('Avergae Delivery Time (days)', round(avg_time,2))

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

st.info("""Semi-Urban areas show the highest delivery delays, suggesting infrastructure or routing inefficiencies. 
Weather conditions like fog and clouds also significantly increase delivery time.""")

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

col_box1, col_box2, col_box3, col_box4 = st.columns([3,3,3,3])

with col_box1: 
    distance = st.number_input("Distance (km)", min_value=0.0, value=10.0)
with col_box2: 
    weather = st.selectbox("Weather", [col.replace("Weather_", "") for col in X.columns if "Weather_" in col])
with col_box3: 
    area = st.selectbox("Area", [col.replace("Area_", "") for col in X.columns if "Area_" in col])
with col_box4: 
    category = st.selectbox("Category", [col.replace("Category_", "") for col in X.columns if "Category_" in col])

def make_prediction(distance, weather, area, category):

    row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    if "Distance_in_km" in row.columns:
        row["Distance_in_km"] = distance

    if f"Weather_{weather}" in row.columns:
        row[f"Weather_{weather}"] = 1

    if f"Area_{area}" in row.columns:
        row[f"Area_{area}"] = 1

    if f"Category_{category}" in row.columns:
        row[f"Category_{category}"] = 1

    return model.predict(row)[0]

# Create Predict Button
if st.button("Predict"):
    result = make_prediction(distance, weather, area, category)

    if result == 1:
        st.error("High Risk: Slow Delivery Expected")
    else:
        st.success("Low Risk: On-Time Delivery Expected")

#Show feature importance
importances = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values(by="importance", ascending=False)

st.header(" What Impacts Delivery Delays Most")
st.bar_chart(importances.set_index("feature"))
