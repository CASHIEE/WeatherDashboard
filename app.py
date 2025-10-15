import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, month, hour, weekofyear
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# -------------------------------
# Initialize PySpark
# -------------------------------
spark = SparkSession.builder.appName("WeatherAnalytics").getOrCreate()

# -------------------------------
# Load Data
# -------------------------------
df = spark.read.csv("weather_data.csv", header=True, inferSchema=True)

# Convert date_time to timestamp and add features
df = df.withColumn("date_time", col("date_time").cast("timestamp")) \
       .withColumn("day", dayofweek(col("date_time"))) \
       .withColumn("month", month(col("date_time"))) \
       .withColumn("hour", hour(col("date_time"))) \
       .withColumn("week", weekofyear(col("date_time")))

# -------------------------------
# Sidebar: Select Location
# -------------------------------
locations = [row['Location'] for row in df.select('Location').distinct().collect()]
location = st.sidebar.selectbox("Select Location", locations)
loc_df = df.filter(df.Location == location)

# Convert to Pandas for plotting
pandas_df = loc_df.toPandas()

# -------------------------------
# Dashboard Title & Metrics
# -------------------------------
st.title(f"Weather Analytics Dashboard - {location}")
st.subheader("Key Metrics")
st.metric("Avg Temperature (°C)", round(pandas_df['Temperature_C'].mean(), 2))
st.metric("Avg Humidity (%)", round(pandas_df['Humidity_pct'].mean(), 2))
st.metric("Avg Precipitation (mm)", round(pandas_df['Precipitation_mm'].mean(), 2))
st.metric("Avg Wind Speed (km/h)", round(pandas_df['Wind_Speed_kmh'].mean(), 2))

# -------------------------------
# Extreme Events
# -------------------------------
st.subheader("Extreme Events")
extreme_events = pandas_df[
    (pandas_df['Temperature_C'] > 35) |
    (pandas_df['Precipitation_mm'] > 50) |
    (pandas_df['Wind_Speed_kmh'] > 80)
]
st.dataframe(extreme_events)

# -------------------------------
# Monthly Temperature Trend
# -------------------------------
st.subheader("Monthly Temperature Trend")
monthly_avg = pandas_df.groupby(pandas_df['date_time'].dt.month)['Temperature_C'].mean().reset_index()
fig, ax = plt.subplots()
sns.lineplot(data=monthly_avg, x='date_time', y='Temperature_C', marker='o', ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Avg Temperature (°C)")
st.pyplot(fig)

# -------------------------------
# Temperature vs Humidity Scatter
# -------------------------------
st.subheader("Temperature vs Humidity")
fig2 = px.scatter(pandas_df, x="Temperature_C", y="Humidity_pct", title="Temperature vs Humidity")
st.plotly_chart(fig2)

# -------------------------------
# Wind Speed Distribution
# -------------------------------
st.subheader("Wind Speed Distribution")
fig3 = px.box(pandas_df, y="Wind_Speed_kmh", title="Wind Speed Distribution")
st.plotly_chart(fig3)

# -------------------------------
# Correlation Heatmap
# -------------------------------
st.subheader("Correlation Heatmap")
corr = pandas_df[['Temperature_C','Humidity_pct','Precipitation_mm','Wind_Speed_kmh']].corr()
fig4, ax4 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)
