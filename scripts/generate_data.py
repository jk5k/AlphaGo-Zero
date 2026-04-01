import json
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.self_play import SelfPlay
from src.model.model_wrapper import ModelWrapper
from src.data.data_manager import DataManager

# 加载配置
with open('config/config.json', 'r') as f:
    config = json.load(f)

# 创建必要的目录
os.makedirs(config['data_dir'], exist_ok=True)
os.makedirs(config['model_dir'], exist_ok=True)
os.makedirs(config['log_dir'], exist_ok=True)

# 初始化模型
model = ModelWrapper(
    board_size=config['board_size'],
    num_filters=config['num_filters'],
    num_res_blocks=config['num_res_blocks']
)

# 显示GPU使用情况
print(f"Using device: {model.device}")

# 初始化自我对弈
self_play = SelfPlay(
    model=model,
    board_size=config['board_size'],
    num_simulations=config['num_simulations'],
    temp=config['temp']
)

# 生成数据
print(f"Generating {config['num_games']} games...")
data = self_play.generate_data(num_games=config['num_games'])

# 保存数据
data_manager = DataManager(data_dir=config['data_dir'])
data_filename = f"self_play_data_{len(os.listdir(config['data_dir'])) + 1}.json"
data_manager.save_data(data, data_filename)

print(f"Data generated and saved to {data_filename}")
print(f"Total samples: {len(data)}")
