from src.model.neural_network import GomokuNet
import torch

class ModelWrapper:
    def __init__(self, board_size=15, num_filters=256, num_res_blocks=10, model_path=None):
        self.board_size = board_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GomokuNet(board_size, num_filters, num_res_blocks).to(self.device)
        
        if model_path:
            self.model.load_model(model_path)
            self.model.to(self.device)
    
    def predict(self, state):
        """预测落子策略和胜率"""
        return self.model.predict(state)
    
    def save_model(self, path):
        """保存模型"""
        self.model.save_model(path)
    
    def load_model(self, path):
        """加载模型"""
        self.model.load_model(path)
        self.model.to(self.device)
    
    def train(self, data_loader, optimizer, epochs=1):
        """训练模型"""
        self.model.train()
        print(f"Training on {self.device}")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in data_loader:
                states, policy_targets, value_targets = batch
                
                # 转换为张量并移到设备上
                states = torch.tensor(states, dtype=torch.float32).unsqueeze(1).to(self.device)
                policy_targets = torch.tensor(policy_targets, dtype=torch.float32).to(self.device)
                value_targets = torch.tensor(value_targets, dtype=torch.float32).unsqueeze(1).to(self.device)
                
                # 前向传播
                policy_pred, value_pred = self.model(states)
                
                # 计算损失
                policy_loss = F.cross_entropy(policy_pred, policy_targets)
                value_loss = F.mse_loss(value_pred, value_targets)
                total_loss_batch = policy_loss + value_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")

# 导入必要的库
import torch.nn.functional as F
