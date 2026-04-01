import math
import random

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # 当前游戏状态
        self.parent = parent  # 父节点
        self.action = action  # 导致当前状态的动作
        self.children = []  # 子节点
        self.visits = 0  # 访问次数
        self.value = 0  # 累积价值
        self.untried_actions = state.get_legal_moves()  # 未尝试的动作
        self.player = state.get_current_player()  # 当前玩家
    
    def select_child(self, c_puct=1.0):
        """选择子节点，使用UCT公式""" 
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                # UCT公式
                exploit = child.value / child.visits
                explore = math.sqrt(2 * math.log(self.visits) / child.visits)
                score = exploit + c_puct * explore
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, game, policy):
        """扩展节点，根据策略选择动作"""
        if not self.untried_actions:
            return None
        
        # 根据策略概率选择动作
        if policy is not None:
            # 将动作映射到策略概率
            action_probs = {}
            for action in self.untried_actions:
                idx = action[0] * game.board_size + action[1]
                action_probs[action] = policy[idx]
            
            # 按概率选择动作
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            action = random.choices(actions, weights=probs)[0]
        else:
            # 随机选择动作
            action = random.choice(self.untried_actions)
        
        # 从untried_actions中移除该动作
        self.untried_actions.remove(action)
        
        # 创建新的游戏状态
        new_game = Gomoku(game.board_size)
        # 复制当前状态
        for i in range(game.board_size):
            for j in range(game.board_size):
                new_game.board[i][j] = game.board[i][j]
        new_game.current_player = game.current_player
        new_game.history = game.history.copy()
        new_game.game_over = game.game_over
        new_game.winner = game.winner
        
        # 执行动作
        new_game.make_move(action)
        
        # 创建新节点
        child = Node(new_game, self, action)
        self.children.append(child)
        return child
    
    def backpropagate(self, result):
        """回溯更新节点价值和访问次数"""
        self.visits += 1
        self.value += result
        
        if self.parent:
            # 对父节点使用相反的结果
            self.parent.backpropagate(-result)

class MCTS:
    def __init__(self, game, model=None, num_simulations=1000, c_puct=1.0):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(self, state):
        """执行MCTS搜索"""
        root = Node(state)
        
        for _ in range(self.num_simulations):
            node = root
            game_copy = Gomoku(self.game.board_size)
            # 复制当前状态
            for i in range(self.game.board_size):
                for j in range(self.game.board_size):
                    game_copy.board[i][j] = state.board[i][j]
            game_copy.current_player = state.current_player
            game_copy.history = state.history.copy()
            game_copy.game_over = state.game_over
            game_copy.winner = state.winner
            
            # 选择阶段
            while not node.untried_actions and node.children:
                node = node.select_child(self.c_puct)
                game_copy.make_move(node.action)
            
            # 扩展阶段
            if not game_copy.game_over:
                policy = None
                if self.model:
                    # 使用神经网络预测策略和价值
                    state_input = game_copy.get_state()
                    policy, _ = self.model.predict(state_input)
                
                child = node.expand(game_copy, policy)
                if child:
                    node = child
                    game_copy.make_move(child.action)
            
            # 模拟阶段
            result = self._rollout(game_copy)
            
            # 回溯阶段
            node.backpropagate(result)
        
        # 返回根节点的最佳动作
        return self._get_best_action(root)
    
    def _rollout(self, game):
        """执行随机模拟"""
        game_copy = Gomoku(game.board_size)
        # 复制当前状态
        for i in range(game.board_size):
            for j in range(game.board_size):
                game_copy.board[i][j] = game.board[i][j]
        game_copy.current_player = game.current_player
        game_copy.history = game.history.copy()
        game_copy.game_over = game.game_over
        game_copy.winner = game.winner
        
        while not game_copy.game_over:
            legal_moves = game_copy.get_legal_moves()
            if not legal_moves:
                break
            
            move = random.choice(legal_moves)
            game_copy.make_move(move)
        
        return game_copy.winner
    
    def _get_best_action(self, root):
        """根据访问次数选择最佳动作"""
        best_visits = -1
        best_action = None
        
        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = child.action
        
        return best_action
    
    def get_action_probs(self, state, temp=1e-3):
        """获取动作概率分布"""
        root = Node(state)
        
        for _ in range(self.num_simulations):
            node = root
            game_copy = Gomoku(self.game.board_size)
            # 复制当前状态
            for i in range(self.game.board_size):
                for j in range(self.game.board_size):
                    game_copy.board[i][j] = state.board[i][j]
            game_copy.current_player = state.current_player
            game_copy.history = state.history.copy()
            game_copy.game_over = state.game_over
            game_copy.winner = state.winner
            
            # 选择阶段
            while not node.untried_actions and node.children:
                node = node.select_child(self.c_puct)
                game_copy.make_move(node.action)
            
            # 扩展阶段
            if not game_copy.game_over:
                policy = None
                if self.model:
                    # 使用神经网络预测策略和价值
                    state_input = game_copy.get_state()
                    policy, _ = self.model.predict(state_input)
                
                child = node.expand(game_copy, policy)
                if child:
                    node = child
                    game_copy.make_move(child.action)
            
            # 模拟阶段
            result = self._rollout(game_copy)
            
            # 回溯阶段
            node.backpropagate(result)
        
        # 计算动作概率
        action_probs = {}
        total_visits = sum(child.visits for child in root.children)
        
        for child in root.children:
            if temp == 0:
                # 温度为0时，选择访问次数最多的动作
                if child.visits == max(c.visits for c in root.children):
                    action_probs[child.action] = 1.0
                else:
                    action_probs[child.action] = 0.0
            else:
                # 温度不为0时，使用softmax
                action_probs[child.action] = (child.visits ** (1/temp)) / (total_visits ** (1/temp))
        
        return action_probs

# 导入Gomoku类
from src.game.game import Gomoku
