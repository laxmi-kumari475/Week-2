#import necessary libraries
import numpy as np #numerical python - linear algebra
import pandas as pd #data manipulation

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

#load the dataset
df = pd.read_csv(r'Dataset.csv',sep=';')
print(df)

#dataset info
df.info()

#rows and cols
print(df.shape)

#Statistics of the data
print(df.describe().T)

#Missing values
print(df.isnull().sum())

#date is in object-date format
df['date'] = pd.to_datetime(df['date'], format = '%d.%m.%Y')
print(df)

#again info
df.info()

#sorting values
df = df.sort_values(by=['id','date'])
print(df.head())

# data on basis of month and year
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
print(df.head())

#columns
print(df.columns)

#accesing pollutants
pollutants=['O2','NO3','NO2','SO4','PO4','CL']
print(pollutants)


