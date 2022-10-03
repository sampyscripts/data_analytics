
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


citi_bikes = pd.ExcelFile("NYCiti_Bikes.xlsx")
citi_bikes.sheet_names #Checking sheet names present in citi_bikes workbook 

raw_bike_records = citi_bikes.parse(sheet_name = "NYCitiBikes")

#--- DATA CLEANING ---
#raw_bike_records.info()
#----To fill missing values (missing completely at random)
excluding_missing_values = pd.DataFrame(
    raw_bike_records.loc[raw_bike_records["End Station Name"].notna(), ["End Station ID", "End Station Name"]],
    )
excluding_missing_values = excluding_missing_values.set_index("End Station ID")
#dropping duplicates  from the dataframe
excluding_missing_values = excluding_missing_values.drop_duplicates()

df = pd.DataFrame({
    "End Station ID": raw_bike_records["End Station ID"].values,
    "End Station Names": ""
    })

df = df.set_index("End Station ID")

#Mapping excluding_missing_values to df to give missing values of raw_bike_records["End Station Name"]
raw_bike_records["End Station Name"] = (excluding_missing_values.index.map(df["End Station Name"])).array

#----Dropping duplicates from Citi Bike data
raw_bike_records = raw_bike_records.drop_duplicates()

#----Outliers
#Checking for outliers using descriptive analysis
raw_bike_records.describe()[["Age", "Trip_Duration_in_min"]]

#Checking for outliers using pandas boxplot
raw_bike_records["Age"].plot.box()
raw_bike_records["Trip_Duration_in_min"].plot.box()

#Removing & capping outliers
'''Removing infeasible data (ages greater than 100)'''
bike_records = raw_bike_records[raw_bike_records["Age"]<100]

'''Capping outliers to 15000 (app 10 days)'''
raw_bike_records["Trip_Duration_in_min"][raw_bike_records["Trip_Duration_in_min"]>15000] = 15000


#---------------------------------------------------------
#--- ANSWERS TO QUESTIONS ---
#Q1--What are the most popular pick-up locations across the city for Citi Bike rental? 
start_station_frequency = bike_records.groupby(["Start Station Name"])["Bike ID"].count()
top_10 = start_station_frequency.to_frame().sort_values(ascending=False)[:10]
fig = plt.figure(figsize=(8,5))
ax = fig.add_axes([0.2,0.2,0.6,0.6])
ax.barh(top_10.index, top_10["Bike ID"], 0.7) #making an horizontal bar chart

#Q2--How does the average trip duration vary across different age groups, and over time?
#A
trip_age_group = bike_records.groupby(["Age Groups"])["Trip_Duration_in_min"].mean().round()
trip_by_age = trip_age_group.sort_values(ascending=False).to_frame()
ax.bar(trip_by_age.index, trip_by_age["Trip_Duration_in_min"], 0.7) #making a bar chart

#B
avg_monthly_trip = bike_records.groupby(["Month"])["Trip_Duration_in_min"].mean().round()
framed_monthly_trip = avg_monthly_trip.to_frame()
ax.plot(framed_monthly_trip.index, framed_monthly_trip["Trip_Duration_in_min"]) #making a plot

#Q3--What time of the year does Citi Bike record the highest bike demand?
rent_per_season = bike_records.groupby(["Month"])["Bike ID"].count()
rent_per_season = rent_per_season.to_frame()
ax.plot(rent_per_season.index, rent_per_season["Bike ID"]) #making a plot

#Q4--Which age group rents the most bikes?
rental_per_ageGroup = bike_records.groupby(["Age Groups"])["Bike ID"].count()
rental_per_ageGroup = rental_per_ageGroup.sort_values().to_frame()
ax.barh(rental_per_ageGroup.index, rental_per_ageGroup["Bike ID"]) #making an horizontal bar chart

#Q5--How does bike rental vary across the two user groups (one-time users vs long-term subscribers) on different days of the week?
weekly_user_type = bike_records.pivot_table(index = "Weekday", columns = "User Type", values = "Bike ID", aggfunc = "count")
weekly_user_type.plot.area() #making an area plot

#Q6--Do factors like weather and user age impact the average bike trip duration?
#A--Weather impacting trip duration?
weather_ave_trip = bike_records.groupby(["Temperature"])["Trip_Duration_in_min"].mean().round()
df_weather_trip = weather_ave_trip.to_frame()
#Correlation Estimate
correlation_table = pd.DataFrame({
    "Temp": df_weather_trip.index, "Average Trip": df_weather_trip["Trip_Duration_in_min"]
    })
correlation_table["Temp"].corr(correlation_table["Average Trip"])
ax.scatter(corr_table["Temp"], corr_table["Average Trip"]) #making a scatter plot

#B--Age impacting trip duration?
age_ave_trip = bike_records.groupby(["Age"])["Trip_Duration_in_min"].mean().round()
df_age_trip = age_ave_trip.to_frame()
#Correlation Estimate
corr_table = pd.DataFrame({
    "Age": df_age_trip.index, "Average Trip": df_age_trip["Trip_Duration_in_min"]
    })
corr_table["Age"].corr(corr_table["Average Trip"]) 
ax.scatter(corr_table["Age"], corr_table["Average Trip"]) #making a scatter plot
