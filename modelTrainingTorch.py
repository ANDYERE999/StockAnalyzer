import pandas as pd
import numpy as np
import glob
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split  # 导入 train_test_split

# 检查 CUDA 是否可用
def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch is using GPU.")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available. PyTorch is using CPU.")

check_cuda()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Fill missing values
df[features] = df[features].fillna(method='ffill')  # 使用前一天的数据填充缺失值

# Split data into training and testing sets
X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and move to device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the model
class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(len(features), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = StockPredictor().to(device)  # 将模型移动到设备

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
print("Training Progress:")
for epoch in tqdm(range(100), desc="Training StockPredictor"):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # 将数据移动到设备
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    mse = criterion(y_pred, y_test_tensor).item()
print(f'Mean Squared Error: {mse}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

# Define prediction function
def predict_stock_value(stock_code, date):
    # Filter data for the given stock code and date
    stock_data = df[(df['code'] == stock_code) & (df['date'] == pd.to_datetime(date))]

    if stock_data.empty:
        print("No data available for the given stock code and date.")
        return

    # Extract features
    new_data = stock_data[features].values
    new_data = pd.DataFrame(new_data, columns=features).fillna(method='ffill').values  # 填充缺失值
    new_data_tensor = torch.tensor(new_data, dtype=torch.float32).to(device)

    # Load the trained model
    model = StockPredictor().to(device)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    # Predict
    with torch.no_grad():
        predicted_open = model(new_data_tensor).item()
    print(f'Predicted Open for {stock_code} on {date}: {predicted_open}')

# Example call
predict_stock_value('sh.600000', '2024-12-07')