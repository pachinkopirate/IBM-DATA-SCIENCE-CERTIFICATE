'''
Space X Falcon 9 First Stage Landing Prediction
Data Wrangling

Objectives
Perform exploratory Data Analysis and determine Training Labels

Exploratory Data Analysis
Determine Training Labels
'''
# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np

#load data
df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
df.head(10)

#Identify and calculate the percentage of the missing values in each attribute
df.isnull().sum()/df.count()*100

#Identify which columns are numerical and categorical:
df.dtypes

'''
TASK 1: Calculate the number of launches on each site
'''

# Apply value_counts() on column LaunchSite
df['LaunchSite'].value_counts()

'''
TASK 2: Calculate the number and occurrence of each orbit
'''
# Apply value_counts on Orbit column
df['Orbit'].value_counts()

'''
TASK 3: Calculate the number and occurence of mission outcome per orbit type
'''

# landing_outcomes = values on Outcome column
landing_outcomes = df['Outcome'].value_counts()
landing_outcomes

for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)

bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes

'''
TASK 4: Create a landing outcome label from Outcome column
'''

landing_class_list = []
for outcome in df['Outcome']:
    if outcome in bad_outcomes:
        landing_class = 0
    else:
        landing_class = 1
    landing_class_list.append(landing_class)


# landing_class = 0 if bad_outcome
# landing_class = 1 otherwise

df['Class']=landing_class_list
df[['Class']].head(8)

df.head(15)

#We can use the following line of code to determine the success rate:
df["Class"].mean()
