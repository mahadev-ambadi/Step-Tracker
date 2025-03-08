import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Load Data Function
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    df.rename(columns={'date': 'Date', 'steps': 'Steps'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format
    df = df.sort_values(by='Date')
    return df

# Identify high/low activity days
def activity_insights(df):
    avg_steps = df['Steps'].mean()
    high_activity = df[df['Steps'] > avg_steps]
    low_activity = df[df['Steps'] < avg_steps]
    return high_activity, low_activity, avg_steps

# Generate Recommendations
def generate_recommendations(df):
    weekday_avg = df.groupby(df['Date'].dt.day_name())['Steps'].mean()
    min_day = weekday_avg.idxmin()
    max_day = weekday_avg.idxmax()
    return f"📌 Try increasing activity on **{min_day}s** and maintaining high activity on **{max_day}s**."

# Predict future steps using Linear Regression
def predict_steps(df):
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    model = LinearRegression()
    X = df[['Days']]
    y = df['Steps']
    model.fit(X, y)
    
    # Predict for the next 7 days
    future_dates = [(df['Date'].max() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    future_steps = model.predict([[df['Days'].max() + i] for i in range(1, 8)])

    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Steps': future_steps.astype(int)})
    return pred_df

# Streamlit UI
st.set_page_config(page_title='Interactive Step Tracker', layout='wide')
st.title("📊 Interactive Step Tracker")

uploaded_file = st.file_uploader("📂 Upload your steps.csv file", type=['csv'])
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ Data Loaded Successfully! Hover over the graph to see values.")
    
    # Main Step Graph
    fig = px.line(df, x='Date', y='Steps', title='📈 Daily Step Count', markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Activity Insights
    st.subheader("📌 Activity Insights")
    high_activity, low_activity, avg_steps = activity_insights(df)

    st.write("### **📊 High Activity Days**")
    st.dataframe(high_activity[['Date', 'Steps']])

    st.write("### **📉 Low Activity Days**")
    st.dataframe(low_activity[['Date', 'Steps']])

    # High/Low Activity Line Graph
    st.write("### **📊 High & Low Activity Days (Linear Graph)**")
    df['Activity Type'] = df['Steps'].apply(lambda x: 'High' if x > avg_steps else 'Low')

    activity_fig = px.line(df, x='Date', y='Steps', color='Activity Type', 
                           title="📈 High & Low Activity Days",
                           labels={'Steps': 'Step Count', 'Date': 'Date'},
                           color_discrete_map={"High": "green", "Low": "red"},
                           markers=True)

    st.plotly_chart(activity_fig, use_container_width=True)

    # Recommendations
    st.subheader("📢 Recommendations")
    st.write(generate_recommendations(df))

    # Weekly Step Prediction
    st.subheader("🔮 Weekly Step Prediction")
    predictions = predict_steps(df)
    st.dataframe(predictions)

    # Prediction Graph
    st.write("### **📈 Step Count Prediction for Next 7 Days**")
    pred_fig = px.line(predictions, x='Date', y='Predicted Steps', markers=True, title="Predicted Steps for Next 7 Days")
    st.plotly_chart(pred_fig, use_container_width=True)

    # Personal Health Monitoring
    st.subheader("🩺 Personal Health Monitoring")
    min_steps = df['Steps'].min()
    max_steps = df['Steps'].max()
    st.write(f"**📊 Average Daily Steps:** {int(avg_steps)}")
    st.write(f"**📉 Minimum Steps in a Day:** {min_steps}")
    st.write(f"**📈 Maximum Steps in a Day:** {max_steps}")

    # Cloud Deployment Note
    st.info("🚀 For real-time tracking, deploy this on a cloud service like AWS, GCP, or Streamlit Cloud.")
else:
    st.error("❌ Error: 'steps.csv' not found. Please upload a valid step count file.")
