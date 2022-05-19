'''
SpaceX Falcon 9 First Stage Landing Prediction¶
Assignment: Exploring and Preparing Data
'''
# andas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns

df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

# If you were unable to complete the previous lab correctly you can uncomment and load this csv

# df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_2.csv')

df.head(5)

sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()

'''
TASK 1: Visualize the relationship between Flight Number and Launch Site
'''
# Plot a scatter point chart with x axis to be Flight Number and y axis to be the launch site, and hue to be the class value
sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()

'''
TASK 2: Visualize the relationship between Payload and Launch Site
'''
# Plot a scatter point chart with x axis to be Pay Load Mass (kg) and y axis to be the launch site, and hue to be the class value
sns.catplot(y="LaunchSite", x="PayloadMass", hue="Class", data=df, aspect = 5)
plt.xlabel("PayloadMass",fontsize=20)
plt.ylabel("LaunchSite",fontsize=20)
plt.show()

'''
TASK 3: Visualize the relationship between success rate of each orbit type
'''
# HINT use groupby method on Orbit column and get the mean of Class column

# sns.barplot(y="Class", x="Orbit", hue="Orbit", data=df)
# plt.xlabel("Orbit type",fontsize=20)
# plt.ylabel("Success",fontsize=20)
# plt.show()

# # perform groupby
success_rate = df.groupby('Orbit').mean()
success_rate.reset_index(inplace=True)
success_rate.shape
sns.barplot(x="Orbit",y="Class",data=success_rate,hue='Orbit')

'''
TASK 4: Visualize the relationship between FlightNumber and Orbit type
'''
# Plot a scatter point chart with x axis to be FlightNumber and y axis to be the Orbit, and hue to be the class value

sns.scatterplot(y="Orbit", x="FlightNumber", hue="Class", data=df)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.show()

'''
TASK 5: Visualize the relationship between Payload and Orbit type
'''
# Plot a scatter point chart with x axis to be Payload and y axis to be the Orbit, and hue to be the class value

sns.scatterplot(y="Orbit", x="PayloadMass", hue="Class", data=df)
plt.xlabel("Payload",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.show()

'''
TASK 6: Visualize the launch success yearly trend
'''

# A function to Extract years from the date
year=[]
def Extract_year(date):
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year

# Plot a line chart with x axis to be the extracted year and y axis to be the success rate
#df['extracted_year'] = Extract_year(df['Date'])
df
#sns.lineplot(x='extracted_year', y = '
success_rate2 = df.groupby('extracted_year').mean()
success_rate2.reset_index(inplace=True)
success_rate2
sns.lineplot(x="extracted_year",y="Class",data=success_rate2)
plt.show()

'''
Features Engineering
'''
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()

'''
TASK 7: Create dummy variables to categorical columns
'''
# HINT: Use get_dummies() function on the categorical columns
features_one_hot = pd.get_dummies(data=features, columns=['Orbit','LaunchSite', 'LandingPad','Serial'])

features_one_hot.head()
features
#features_one_hot.iloc[1]

#features_one_hot['Orbit']

'''
TASK 8: Cast all numeric columns to float64
'''
# HINT: use astype function
features_one_hot.astype(float)
