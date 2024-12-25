import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
import glob
import time
from tqdm import tqdm
from sklearn.externals import joblib

# Read all CSV files and merge them
file_path = './files/db/*.csv'
all_files = glob.glob(file_path)

df_list = []
for file in all_files:
    try:
        df = pd.read_csv(file)
        if df.empty:
            print(f"Skipping empty file: {file}")
            continue
        df_list.append(df)
    except pd.errors.EmptyDataError:
        print(f"Skipping empty file: {file}")

# Merge all data
df = pd.concat(df_list, ignore_index=True)

# Process date format
df['date'] = pd.to_datetime(df['date'])

# Remove weekend data
df = df[df['date'].dt.weekday < 5]

# Select features and target
features = ['preclose', 'high', 'low', 'volume', 'amount']
target = 'open'

# Split data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Show progress bar using tqdm
print("Training Progress:")
for _ in tqdm(range(1), desc="Training RandomForest"):
    model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model
joblib.dump(model, 'trained_model.pkl')

# Define prediction function
def predict_stock_value(stock_code, date):
    # Filter data for the given stock code and date
    stock_data = df[(df['code'] == stock_code) & (df['date'] == pd.to_datetime(date))]

    if stock_data.empty:
        print("No data available for the given stock code and date.")
        return

    # Extract features
    new_data = stock_data[features].values

    # Load the trained model
    model = joblib.load('trained_model.pkl')

    # Predict
    predicted_open = model.predict(new_data)
    print(f'Predicted Open for {stock_code} on {date}: {predicted_open[0]}')

# Example call
predict_stock_value('sh.600000', '2024-12-07')