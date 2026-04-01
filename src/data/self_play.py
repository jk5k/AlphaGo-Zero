import time
import random
from src.game.game import Gomoku
from src.mcts.mcts import MCTS
from src.model.model_wrapper import ModelWrapper

class SelfPlay:
    def __init__(self, model, board_size=15, num_simulations=1000, temp=1.0):
        self.model = model
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.temp = temp
    
    def generate_game(self):
        """生成一局自我对弈游戏"""
        game = Gomoku(self.board_size)
        game_data = []
        
        while not game.get_game_over():
            # 创建MCTS实例
            mcts = MCTS(game, self.model, self.num_simulations)
            
            # 获取动作概率
            action_probs = mcts.get_action_probs(game, self.temp)
            
            # 记录当前状态
            state = game.get_state()
            
            # 将动作概率转换为向量
            probs = [0] * (self.board_size * self.board_size)
            for action, prob in action_probs.items():
                idx = action[0] * self.board_size + action[1]
                probs[idx] = prob
            
            # 选择动作
            actions = list(action_probs.keys())
            action_probs_list = list(action_probs.values())
            action = random.choices(actions, weights=action_probs_list)[0]
            
            # 记录数据
            game_data.append({
                'state': state,
                'probs': probs,
                'player': game.get_current_player()
            })
            
            # 执行动作
            game.make_move(action)
        
        # 计算最终结果
        winner = game.get_winner()
        
        # 更新数据中的价值标签
        for data in game_data:
            if winner == 0:
                data['value'] = 0
            else:
                data['value'] = 1 if data['player'] == winner else -1
        
        return game_data, winner
    
    def generate_data(self, num_games=100):
        """生成指定数量的自我对弈数据"""
        all_data = []
        start_time = time.time()
        
        for i in range(num_games):
            game_data, winner = self.generate_game()
            all_data.extend(game_data)
            
            # 打印进度
            if (i + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                games_per_hour = (i + 1) / (elapsed_time / 3600)
                print(f"Game {i+1}/{num_games}, Winner: {winner}, Games per hour: {games_per_hour:.2f}")
        
        return all_data
