import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class OctomapCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # LSTM网络处理序列数据
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 分层一：速度方向上细化高度-方位角距离矩阵
        self.octomap_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # (B,16,18,36)
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),  # (B,16,9,18)
            
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # (B,32,9,18)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),  # (B,32,4,9)
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # (B,64,4,9)
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Flatten(),
            
            nn.Linear(64 * 4*9, 1024),
            nn.ELU(),

            nn.Linear(1024, 256),
            nn.ELU(),
            nn.LayerNorm(256),
        )
        
        # 分层二：无人机状态信息与任务信息序列
        self.stat_fc = nn.Sequential(
            nn.Linear(15, 32),
            nn.ELU(),

            nn.Linear(32, 64),
            nn.ELU(),

            nn.Linear(64, 64),
            nn.ELU(),
            nn.LayerNorm(64),
        )

        # 融合后全连接层加宽，激活用GELU
        self.combined_fc = nn.Sequential(
            nn.Linear(256+64, 512),
            nn.ELU(),

            nn.Linear(512, 256),
            nn.ELU(),

            nn.Linear(256, features_dim),
            nn.ELU(),
            nn.LayerNorm(features_dim),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 不要在这里处理 device！交给外部调用者处理
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        # 注意：不再 .to(self.device)

        obs_t = x 
                
        # 2. 拆分 OctoMap 和状态信息
        octomap_t = obs_t[:, :18*36].view(-1, 1, 18, 36)  
        stat_t = obs_t[:, -15:]                          
                
        # 3. 分别提取特征
        octomap_features_t = self.octomap_cnn(octomap_t)  
        stat_features_t = self.stat_fc(stat_t)            
                
        # 4. 融合特征
        combined_features_t = torch.cat((octomap_features_t, stat_features_t), dim=1) 
                
        # 5. 映射到最终的特征维度
        final_feature_t = self.combined_fc(combined_features_t) 
        return final_feature_t
    
model = OctomapCNN(observation_space=663, features_dim=256)
# 设置模型为评估模式
model.eval()

# 创建一个与模型输入形状匹配的虚拟输入张量
dummy_input = torch.randn(1, 663)  # 这里的1是batch size

# 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "octomap_cnn.onnx",
    verbose=False,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,  # 推荐用 13 或更高，支持更多算子
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)
print("✅ ONNX 模型已成功导出到 octomap_cnn.onnx")