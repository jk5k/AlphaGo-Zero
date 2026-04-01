import json
import os
import sys
import torch
import torch.optim as optim

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.model_wrapper import ModelWrapper
from src.data.data_manager import DataManager

# 加载配置
with open('config/config.json', 'r') as f:
    config = json.load(f)

# 创建必要的目录
os.makedirs(config['data_dir'], exist_ok=True)
os.makedirs(config['model_dir'], exist_ok=True)
os.makedirs(config['log_dir'], exist_ok=True)

# 初始化数据管理器
data_manager = DataManager(data_dir=config['data_dir'])

# 加载所有数据
data_files = [f for f in os.listdir(config['data_dir']) if f.endswith('.json')]
all_data = []
for file in data_files:
    data = data_manager.load_data(file)
    all_data.extend(data)

print(f"Loaded {len(all_data)} samples from {len(data_files)} files")

if len(all_data) == 0:
    print("Error: No training data found. Please run generate_data.py first to generate training data.")
    exit(1)

# 分割数据
train_data, val_data = data_manager.split_data(all_data, train_ratio=config['train_ratio'])
print(f"Train data: {len(train_data)}, Validation data: {len(val_data)}")

# 创建数据加载器
train_loader = data_manager.create_dataloader(train_data, batch_size=config['batch_size'], shuffle=True)
val_loader = data_manager.create_dataloader(val_data, batch_size=config['batch_size'], shuffle=False)

# 初始化模型
model = ModelWrapper(
    board_size=config['board_size'],
    num_filters=config['num_filters'],
    num_res_blocks=config['num_res_blocks']
)

# 初始化优化器
optimizer = optim.Adam(model.model.parameters(), lr=config['learning_rate'])

# 训练模型
print("Training model...")
model.train(train_loader, optimizer, epochs=config['epochs'])

# 保存模型
model_filename = f"model_{len(os.listdir(config['model_dir'])) + 1}.pt"
model_path = os.path.join(config['model_dir'], model_filename)
model.save_model(model_path)

print(f"Model saved to {model_path}")
