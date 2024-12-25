import os
import pandas as pd
import numpy as np
import glob
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

# 定义特征列
features = ['preclose', 'high', 'low', 'volume', 'amount']

# 定义模型结构
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

# 加载训练好的模型
model = StockPredictor().to(device)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# 定义预测函数
def predict_stock_value(stock_code, date):
    # 读取所有 CSV 文件并合并
    file_path = './files/db/*.csv'  # 修改为你的数据集路径
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

    # 检查是否有有效的 DataFrame
    if not df_list:
        raise ValueError("No valid CSV files found. Please check your dataset.")

    # 合并所有数据
    df = pd.concat(df_list, ignore_index=True)

    # 处理日期格式
    df['date'] = pd.to_datetime(df['date'])

    # 移除周末数据
    df = df[df['date'].dt.weekday < 5]

    # 填充缺失值
    df[features] = df[features].ffill()

    # 筛选给定股票代码和日期的数据
    stock_data = df[(df['code'] == stock_code) & (df['date'] == pd.to_datetime(date))]

    if stock_data.empty:
        print("No data available for the given stock code and date.")
        return

    # 提取特征
    new_data = stock_data[features].values

    # 如果没有前一天的数据，使用均值填充
    if np.isnan(new_data).any():
        new_data = np.nan_to_num(new_data, nan=np.nanmean(new_data))

    new_data_tensor = torch.tensor(new_data, dtype=torch.float32).to(device)

    # 预测
    with torch.no_grad():
        predicted_open = model(new_data_tensor).item()
    print(f'Predicted Open for {stock_code} on {date}: {predicted_open}')

# 示例调用
predict_stock_value('sh.600000', '2024-12-07')