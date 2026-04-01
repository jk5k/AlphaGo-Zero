import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuNet(nn.Module):
    def __init__(self, board_size=15, num_filters=256, num_res_blocks=10):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        
        # 输入层
        self.input_layer = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        
        # 残差块
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(self._create_res_block())
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size),
            nn.Softmax(dim=1)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def _create_res_block(self):
        return nn.Sequential(
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_filters)
        )
    
    def forward(self, x):
        # 输入形状: (batch_size, 1, board_size, board_size)
        x = self.input_layer(x)
        x = F.relu(x)
        
        # 通过残差块
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x += residual
            x = F.relu(x)
        
        # 策略头输出
        policy = self.policy_head(x)
        
        # 价值头输出
        value = self.value_head(x)
        
        return policy, value
    
    def predict(self, state):
        """预测落子策略和胜率"""
        # 转换状态为张量并移到模型所在设备
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        state_tensor = state_tensor.to(next(self.parameters()).device)
        
        # 模型推理
        with torch.no_grad():
            policy, value = self.forward(state_tensor)
        
        # 转换为numpy数组
        policy = policy.squeeze().cpu().numpy()
        value = value.squeeze().item()
        
        return policy, value
    
    def save_model(self, path):
        """保存模型"""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """加载模型"""
        self.load_state_dict(torch.load(path))
        self.eval()
