class Gomoku:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 1  # 1 for black, -1 for white
        self.history = []
        self.game_over = False
        self.winner = 0
    
    def reset(self):
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 1
        self.history = []
        self.game_over = False
        self.winner = 0
    
    def get_board(self):
        return self.board
    
    def get_current_player(self):
        return self.current_player
    
    def get_game_over(self):
        return self.game_over
    
    def get_winner(self):
        return self.winner
    
    def get_legal_moves(self):
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal_moves.append((i, j))
        return legal_moves
    
    def make_move(self, move):
        if self.game_over:
            return False
        
        i, j = move
        if i < 0 or i >= self.board_size or j < 0 or j >= self.board_size:
            return False
        
        if self.board[i][j] != 0:
            return False
        
        self.board[i][j] = self.current_player
        self.history.append(move)
        
        if self._check_win(move):
            self.game_over = True
            self.winner = self.current_player
        elif len(self.history) == self.board_size * self.board_size:
            self.game_over = True
            self.winner = 0  # draw
        
        self.current_player *= -1
        return True
    
    def _check_win(self, move):
        i, j = move
        player = self.board[i][j]
        
        # Check horizontal
        count = 1
        for k in range(1, 5):
            if j + k < self.board_size and self.board[i][j + k] == player:
                count += 1
            else:
                break
        for k in range(1, 5):
            if j - k >= 0 and self.board[i][j - k] == player:
                count += 1
            else:
                break
        if count >= 5:
            return True
        
        # Check vertical
        count = 1
        for k in range(1, 5):
            if i + k < self.board_size and self.board[i + k][j] == player:
                count += 1
            else:
                break
        for k in range(1, 5):
            if i - k >= 0 and self.board[i - k][j] == player:
                count += 1
            else:
                break
        if count >= 5:
            return True
        
        # Check diagonal (top-left to bottom-right)
        count = 1
        for k in range(1, 5):
            if i + k < self.board_size and j + k < self.board_size and self.board[i + k][j + k] == player:
                count += 1
            else:
                break
        for k in range(1, 5):
            if i - k >= 0 and j - k >= 0 and self.board[i - k][j - k] == player:
                count += 1
            else:
                break
        if count >= 5:
            return True
        
        # Check diagonal (bottom-left to top-right)
        count = 1
        for k in range(1, 5):
            if i + k < self.board_size and j - k >= 0 and self.board[i + k][j - k] == player:
                count += 1
            else:
                break
        for k in range(1, 5):
            if i - k >= 0 and j + k < self.board_size and self.board[i - k][j + k] == player:
                count += 1
            else:
                break
        if count >= 5:
            return True
        
        return False
    
    def get_state(self):
        """返回当前游戏状态，用于神经网络输入"""
        state = []
        for i in range(self.board_size):
            row = []
            for j in range(self.board_size):
                if self.board[i][j] == 1:
                    row.append(1.0)
                elif self.board[i][j] == -1:
                    row.append(-1.0)
                else:
                    row.append(0.0)
            state.append(row)
        return state
    
    def undo_move(self):
        if not self.history:
            return False
        
        move = self.history.pop()
        i, j = move
        self.board[i][j] = 0
        self.current_player *= -1
        self.game_over = False
        self.winner = 0
        return True
    
    def print_board(self):
        print("  " + " ".join(str(i) for i in range(self.board_size)))
        for i in range(self.board_size):
            row = [str(i)]
            for j in range(self.board_size):
                if self.board[i][j] == 1:
                    row.append("X")
                elif self.board[i][j] == -1:
                    row.append("O")
                else:
                    row.append(".")
            print(" ".join(row))
