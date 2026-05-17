# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# PAGE CONFIG
st.set_page_config(
    page_title="Logistics ML Dashboard",
    layout="wide"
)


# LOAD DATA

df = pd.read_csv("output/amazon_model_dataset.csv")
clean_df = pd.read_csv("output/amazon_cleaned_data.csv")


# MODEL TRAINING
X = df.drop("Slow_Delivery", axis=1)
y = df["Slow_Delivery"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# TABS

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    " EDA",
    "Model Performance",
    "Prediction Tool"
])

# TAB 1: OVERVIEW
with tab1:

    st.title("Logistics Delivery Performance Dashboard")

    st.markdown("""
    **Business Objective:**  
    Predict and reduce delivery delays using machine learning.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Deliveries", len(df))
    col2.metric("Slow Delivery Rate", f"{df['Slow_Delivery'].mean()*100:.2f}%")
    col3.metric("Avg Delivery Time", f"{clean_df['Delivery_Time'].mean():.2f}")

# TAB 2: EDA
with tab2:

    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Delivery Time by Area")
        area_data = clean_df.groupby("Area")["Delivery_Time"].median()

        fig, ax = plt.subplots()
        area_data.plot(kind="bar", ax=ax)
        ax.set_ylabel("Median Delivery Time")
        st.pyplot(fig)

    with col2:
        st.subheader("Delivery Time by Weather")
        weather_data = clean_df.groupby("Weather")["Delivery_Time"].median()

        fig, ax = plt.subplots()
        weather_data.plot(kind="bar", ax=ax)
        ax.set_ylabel("Median Delivery Time")
        st.pyplot(fig)

    st.info("""
    Semi-Urban areas show the highest delays.
    Weather like fog and clouds increase delivery time significantly.
    """)

# TAB 3: MODEL PERFORMANCE
with tab3:

    st.title(" Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Metrics")

        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        st.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
        st.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
        st.metric("F1 Score", f"{f1_score(y_test, y_pred):.3f}")

    st.subheader("Classification Report")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Feature Importance")

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    st.bar_chart(importances.set_index("feature"))

# TAB 4: PREDICTION
with tab4:

    st.title(" Predict Delivery Delay Risk")

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

    def make_prediction(distance, weather, area, category):

        row = pd.DataFrame(columns=X.columns)
        row.loc[0] = 0

        if "Distance_in_km" in row.columns:
            row["Distance_in_km"] = distance

        if f"Weather_{weather}" in row.columns:
            row[f"Weather_{weather}"] = 1

        if f"Area_{area}" in row.columns:
            row[f"Area_{area}"] = 1

        if f"Category_{category}" in row.columns:
            row[f"Category_{category}"] = 1

        return model.predict(row)[0]

    if st.button("Predict"):
        result = make_prediction(distance, weather, area, category)

        if result == 1:
            st.error("⚠ High Risk: Slow Delivery Expected")
        else:
            st.success("Low Risk: On-Time Delivery Expected")


