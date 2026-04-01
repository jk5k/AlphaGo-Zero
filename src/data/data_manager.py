import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GomokuDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        state = item['state']
        probs = item['probs']
        value = item['value']
        return state, probs, value

class DataManager:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_data(self, data, filename):
        """保存数据到文件"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_data(self, filename):
        """从文件加载数据"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    
    def create_dataloader(self, data, batch_size=32, shuffle=True):
        """创建数据加载器"""
        dataset = GomokuDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
    
    def preprocess_data(self, data):
        """预处理数据"""
        # 这里可以添加数据预处理逻辑，例如数据增强、归一化等
        return data
    
    def combine_data(self, data_list):
        """合并多个数据文件"""
        combined_data = []
        for data in data_list:
            combined_data.extend(data)
        return combined_data
    
    def split_data(self, data, train_ratio=0.8):
        """分割数据为训练集和验证集"""
        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        return train_data, val_data

# 导入必要的库
import random
