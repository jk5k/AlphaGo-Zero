import json
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.game import Gomoku
from src.mcts.mcts import MCTS
from src.model.model_wrapper import ModelWrapper
from src.data.self_play import SelfPlay

# 加载配置
with open('config/config.json', 'r') as f:
    config = json.load(f)

print("Testing Gomoku game...")
# 测试游戏逻辑
game = Gomoku(config['board_size'])
game.print_board()

# 测试落子
move = (7, 7)
game.make_move(move)
game.print_board()

# 测试获取合法移动
legal_moves = game.get_legal_moves()
print(f"Legal moves: {len(legal_moves)}")

print("\nTesting MCTS...")
# 测试MCTS
model = ModelWrapper(
    board_size=config['board_size'],
    num_filters=config['num_filters'],
    num_res_blocks=config['num_res_blocks']
)

mcts = MCTS(game, model, num_simulations=100)
action = mcts.search(game)
print(f"MCTS selected action: {action}")

print("\nTesting self-play...")
# 测试自我对弈
self_play = SelfPlay(
    model=model,
    board_size=config['board_size'],
    num_simulations=100,
    temp=config['temp']
)

game_data, winner = self_play.generate_game()
print(f"Game completed, winner: {winner}")
print(f"Generated {len(game_data)} training samples")

print("\nSystem test completed successfully!")
