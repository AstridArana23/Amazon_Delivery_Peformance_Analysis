## Overview 
This project analyzes logistics delivery performance data to identify key factors affecting delivery speed, efficiency, and agent ratings. 

By combining Python and ML, the goal is to uncover operational bottlenecks and provide data-driven recommendations to improve delivery reliability and reduce delays.  

## Business Problem
Logistics companies often face challenges such as late deliveries, inconsistent carrier performance, and regional inefficiencies. This project hopes to answer: 
- What factors contribute the most to delivery delays (distance, areas, weather)?
- Which areas or categories perform best or worst?
- How do delivery delays impact Agent Ratings?
- Can we predict whether a delivery will be slow?

## 📁 Project Structure 
- notebooks/ -> Python + Google Colab
- data/ → raw dataset

## 📈 Dataset 
The dataset contains logistics shipment records with features such as: 
- Order_ID
- Agent_Age
- Agent_Rating
- Store_Latitude
- Store_Longitude
- Drop_Latitude
- Drop_Longitude
- Order_Date
- Order_Time
- Pickup_Time
- Weather
- Traffic
- Vehicle	Area
- Delivery_Time	Category

Source: Amazon Delivery Dataset | Kaggle

## 🔨 Tools & Technologies 
- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Google Colab

## Workflow
This project follows a structured end-to-end data science pipeline to ensure reproducibility and clear business alignment.
1. Problem Definition
2. Data Collection
3. Data Cleaning & Wrangling
4. Exploratory Data Analysis (EDA)
6. Predictive Modeling using machine learning models
7. Conclusion


## Recommendations 
Overall, our analysis shows that delivery performance is shaped by a combination of geographic and environmental factors, and these patterns directly influence agent ratings. 

Semi‑Urban areas consistently produce the longest delivery times, often exceeding 180 minutes, the point at which we begin to see agent ratings decline. This makes Semi‑Urban regions the most critical areas to target for operational improvements. 

Weather conditions further reinforce this pattern. Cloudy and foggy conditions lead to the slowest deliveries, suggesting that visibility challenges have a stronger impact on efficiency than more severe but less frequent conditions such as storms or sandstorms.

Our machine‑learning model supports these findings by identifying Area, Weather, and Distance as key predictors of slow deliveries. The model successfully classifies high‑risk deliveries, allowing us to anticipate delays before they occur. 

Together, these insights highlight where interventions, such as route optimization, resource allocation, or targeted training,  can have the greatest impact. 

By focusing on Semi‑Urban regions and visibility‑related weather conditions, organizations can meaningfully improve both delivery times and agent ratings.

## Author 
Astrid Arana Rivera

⚙️ Aspiring Data Scientist

