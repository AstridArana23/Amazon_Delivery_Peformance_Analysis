# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# LOAD DATA
df = pd.read_csv("output/amazon_model_dataset.csv")
clean_df = pd.read_csv("output/amazon_cleaned_data.csv")

# TITLE
st.title("Amazon Delivery Performance Dashboard")

st.markdown("""
### Business Problem: 
Identify factors contributing to delivery delays and predict high-risk deliveries.

### Goal: 
Improve logistics efficiency and reduce late deliveries.
""")

# KPI SECTION
st.header("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Deliveries", len(df))
col2.metric("Slow Delivery Rate (%)", round(df["Slow_Delivery"].mean() * 100, 2))
col3.metric("Avg Delivery Time", round(clean_df["Delivery_Time"].mean(), 2))

# EDA SECTION
st.header("Exploratory Data Analysis")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    area_data = clean_df.groupby("Area")["Delivery_Time"].median()

    fig, ax = plt.subplots()
    area_data.plot(kind="bar", ax=ax)
    ax.set_title("Median Delivery Time by Area")
    ax.set_ylabel("Delivery Time")
    st.pyplot(fig)

with col_chart2:
    weather_data = clean_df.groupby("Weather")["Delivery_Time"].median()

    fig, ax = plt.subplots()
    weather_data.plot(kind="bar", ax=ax)
    ax.set_title("Median Delivery Time by Weather")
    ax.set_ylabel("Delivery Time")
    st.pyplot(fig)

st.info("""
Semi-Urban areas show the highest delivery delays.
Weather like fog and clouds increases delivery time significantly.
""")

# MODEL TRAINING
st.header("Model Training")

X = df.drop("Slow_Delivery", axis=1)
y = df["Slow_Delivery"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# MODEL EVALUATION
st.header("Model Evaluation")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
st.pyplot(fig)

report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# FEATURE IMPORTANCE
st.header("Key Drivers of Delivery Delays")

importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

st.bar_chart(importances.set_index("feature"))

st.markdown("""
Distance and area type are the strongest predictors of delivery delays.
""")
# PREDICTION SECTION
st.header("Predict Delivery Delay Risk")

st.subheader("Enter Delivery Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    distance = st.number_input("Distance (km)", min_value=0.0, value=10.0)

with col2:
    weather = st.selectbox(
        "Weather",
        [c.replace("Weather_", "") for c in X.columns if "Weather_" in c]
    )

with col3:
    area = st.selectbox(
        "Area",
        [c.replace("Area_", "") for c in X.columns if "Area_" in c]
    )

with col4:
    category = st.selectbox(
        "Category",
        [c.replace("Category_", "") for c in X.columns if "Category_" in c]
    )

# PREDICTION FUNCTION
def make_prediction(distance, weather, area, category):

    row = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)

    if "Distance_in_km" in row.columns:
        row["Distance_in_km"] = distance

    if f"Weather_{weather}" in row.columns:
        row[f"Weather_{weather}"] = 1

    if f"Area_{area}" in row.columns:
        row[f"Area_{area}"] = 1

    if f"Category_{category}" in row.columns:
        row[f"Category_{category}"] = 1

    return model.predict(row)[0]

# OUTPUT
if st.button("Predict"):
    result = make_prediction(distance, weather, area, category)

    if result == 1:
        st.error("High Risk: Slow Delivery Expected")
    else:
        st.success("Low Risk: On-Time Delivery Expected")
