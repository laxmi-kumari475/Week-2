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
print()

#dataset info
df.info()
print()

#rows and cols
print(df.shape)
print()

#Statistics of the data
print(df.describe().T)
print()

#Missing values
print(df.isnull().sum())
print()

#date is in object-date format
df['date'] = pd.to_datetime(df['date'], format = '%d.%m.%Y')
print(df)
print()

#again info
df.info()
print()

#sorting values
df = df.sort_values(by=['id','date'])
print(df.head())
print()

# data on basis of month and year
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
print(df.head())
print()

#columns
print(df.columns)
print()

#accesing pollutants
pollutants=['O2','NO3','NO2','SO4','PO4','CL']

#drop the missing values - dropna
df = df.dropna(subset=pollutants)
print(df.head())
print()

#finding total sum missing values
print(df.isnull().sum())
print()

# Feature and target selection - Feature - independent variable and Target dependent variable
X = df[['id', 'year']]
y = df[pollutants]

# Encoding - onehotencoder - 22 stations - 1 - 1
X_encoded = pd.get_dummies(X, columns=['id'], drop_first=True)
# Train, Test and Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
# Train the model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Performance on the Test Data:")
for i, pollutant in enumerate(pollutants):
    print(f'{pollutant}:')
    print('   MSE:', mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    print('   R2:', r2_score(y_test.iloc[:, i], y_pred[:, i]))
    print()


    station_id = '22'
year_input = 2024

input_data = pd.DataFrame({'year': [year_input], 'id': [station_id]})
input_encoded = pd.get_dummies(input_data, columns=['id'])

# Align with training feature columns
missing_cols = set(X_encoded.columns) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]  # reorder columns

# Predict pollutants
predicted_pollutants = model.predict(input_encoded)[0]

print(f"\nPredicted pollutant levels for station '{station_id}' in {year_input}:")
for p, val in zip(pollutants, predicted_pollutants):
    print(f"  {p}: {val:.2f}")


import joblib

joblib.dump(model, 'pollution_model.pkl')
joblib.dump(X_encoded.columns.tolist(), "model_columns.pkl")
print('Model and cols structure are saved!')
#Model and cols str are saved!

