# Part 1: House Price Prediction Using Linear Regression

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Load the CSV file
df = pd.read_csv("kc_house_data.csv")

# Keep only the columns needed for this assignment
df = df[['sqft_living', 'price']].copy()

# Rename sqft_living so it matches the starter-code format
df.rename(columns={'sqft_living': 'square_footage'}, inplace=True)

# Remove rows with missing values
df = df.dropna()

# Generate location categories
# random_state / seed makes the same locations appear each time
np.random.seed(42)

df['location'] = np.random.choice(
    ['Downtown', 'Suburb', 'Rural'],
    size=len(df),
    p=[0.35, 0.45, 0.20]
)

# Display basic information about the dataset

print("Number of records:", len(df))
print("\nFirst 5 rows:")
print(df.head())

# FEATURES AND TARGET

X = df[['square_footage', 'location']]
y = df['price']

# PREPROCESSING
# OneHotEncoder converts location into numerical values

preprocessor = ColumnTransformer(
    transformers=[
        (
            'location',
            OneHotEncoder(
                drop='first',
                handle_unknown='ignore',
                sparse_output=False
            ),
            ['location']
        )
    ],
    remainder='passthrough'
)

# CREATE LINEAR REGRESSION PIPELINE

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# SPLIT DATA INTO TRAINING AND TESTING DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# TRAIN MODEL

model.fit(X_train, y_train)

# TEST MODEL PERFORMANCE

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R-squared Score: {r2:.3f}")

# PREDICT PRICE OF NEW HOUSE
# Required example: 2000 sq ft in Downtown

new_house = pd.DataFrame({
    'square_footage': [2000],
    'location': ['Downtown']
})

predicted_price = model.predict(new_house)

print(
    f"\nPredicted price for a 2000 sq ft house in Downtown: "
    f"${predicted_price[0]:,.2f}"
)

# DISPLAY MODEL COEFFICIENTS

location_features = (
    model.named_steps['preprocessor']
    .named_transformers_['location']
    .get_feature_names_out(['location'])
).tolist()

feature_names = location_features + ['square_footage']

coefficients = model.named_steps['regressor'].coef_

print("\nModel Coefficients:")

for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: ${coef:,.2f}")

# EXPLAIN SQUARE FOOTAGE COEFFICIENT

sqft_index = feature_names.index('square_footage')
sqft_coefficient = coefficients[sqft_index]

print("\nCoefficient Explanation:")

print(
    f"For each additional square foot, the model predicts "
    f"the house price will change by approximately "
    f"${sqft_coefficient:,.2f}, while location stays the same."
)

print(
    "The location coefficients show how predicted prices differ "
    "between location categories after accounting for square footage."
)
