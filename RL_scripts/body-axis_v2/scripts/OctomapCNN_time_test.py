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
            nn.Linear(8, 32),
            nn.ELU(),

            nn.Linear(32, 64),
            nn.ELU(),

            nn.Linear(64, 96),
            nn.ELU(),
            nn.LayerNorm(96),
        )

        # 融合后全连接层加宽，激活用GELU
        self.combined_fc = nn.Sequential(
            nn.Linear(256+96, 512),
            nn.ELU(),

            nn.Linear(512, 512),
            nn.ELU(),

            nn.Linear(512, features_dim),
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
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        obs_t = x 
            
        # 2. 拆分 OctoMap 和状态信息
        octomap_t = obs_t[:, :18*36].view(-1, 1, 18, 36)  
        stat_t = obs_t[:, -8:]                          
            
        # 3. 分别提取特征
        octomap_features_t = self.octomap_cnn(octomap_t)  
        stat_features_t = self.stat_fc(stat_t)            
            
        # 4. 融合特征
        combined_features_t = torch.cat((octomap_features_t, stat_features_t), dim=1) 
            
        # 5. 映射到最终的特征维度
        final_feature_t = self.combined_fc(combined_features_t) 
        return final_feature_t

import time
from torch.profiler import profile, record_function, ProfilerActivity

def benchmark_octomap_cnn(model, input_shape=(1, 18*36 + 8), num_warmup=100, num_test=1000, use_profiler=False):
    """
    对 OctomapCNN 模型在 GPU 上进行性能基准测试。

    Args:
        model (torch.nn.Module): 已加载到 GPU 的 OctomapCNN 模型实例。
        input_shape (tuple): 输入张量的形状，默认为 (batch_size, feature_dim)。
        num_warmup (int): 预热迭代次数。
        num_test (int): 测试迭代次数。
        use_profiler (bool): 是否使用 PyTorch Profiler 进行详细分析（可选）。

    Returns:
        float: 平均推理时间（毫秒）。
        dict: 包含最小、最大、平均时间及其他统计信息的字典。
    """
    if not next(model.parameters()).is_cuda:
        raise ValueError("模型必须位于 CUDA 设备上进行 GPU 性能测试。")

    device = next(model.parameters()).device
    
    # 生成随机测试输入
    dummy_input = torch.randn(input_shape, device=device, dtype=torch.float32)
    
    # --- 预热阶段 ---
    print(f"开始预热... ({num_warmup} iterations)")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
        torch.cuda.synchronize(device=device)  # 确保预热操作完成
    print("预热完成。")

    # --- 性能测试 ---
    print(f"开始性能测试... ({num_test} iterations)")
    times = []
    
    if use_profiler:
        print("\n--- 开启 PyTorch Profiler ---")
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with torch.no_grad():
                for _ in range(num_test):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    _ = model(dummy_input)
                    end_event.record()
                    
                    torch.cuda.synchronize(device=device)
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    times.append(elapsed_time_ms)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print("--- Profiler 结束 ---\n")

    else:
        # 标准性能测试
        with torch.no_grad():
            for _ in range(num_test):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                
                # 同步以获取准确的事件间隔时间
                torch.cuda.synchronize(device=device)
                elapsed_time_ms = start_event.elapsed_time(end_event)
                times.append(elapsed_time_ms)

    # --- 结果统计 ---
    import numpy as np
    times_np = np.array(times)
    
    stats = {
        "min_time_ms": times_np.min(),
        "max_time_ms": times_np.max(),
        "mean_time_ms": times_np.mean(),
        "std_time_ms": times_np.std(),
        "p50_time_ms": np.percentile(times_np, 50),
        "p95_time_ms": np.percentile(times_np, 95),
        "p99_time_ms": np.percentile(times_np, 99),
    }
    
    print("\n=== 性能测试结果 ===")
    print(f"模型: {model.__class__.__name__}")
    print(f"输入形状: {input_shape}")
    print(f"测试次数: {num_test}")
    print(f"平均推理时间: {stats['mean_time_ms']:.4f} ± {stats['std_time_ms']:.4f} ms")
    print(f"最小推理时间: {stats['min_time_ms']:.4f} ms")
    print(f"最大推理时间: {stats['max_time_ms']:.4f} ms")
    print(f"中位数 (P50): {stats['p50_time_ms']:.4f} ms")
    print(f"P95 推理时间: {stats['p95_time_ms']:.4f} ms")
    print(f"P99 推理时间: {stats['p99_time_ms']:.4f} ms")
    print("==================")
    
    return stats['mean_time_ms'], stats

# --- 使用示例 ---
if __name__ == "__main__":

    # 1. 实例化模型
    # 注意：features_dim 可以根据您的实际需求调整
    model_instance = OctomapCNN(observation_space=None, features_dim=256) 
    
    # 2. 将模型移动到 GPU
    model_instance = model_instance.cuda() # 或 model_instance.to('cuda:0')
    model_instance.eval() # 设置为评估模式

    # 3. 运行基准测试
    mean_time, detailed_stats = benchmark_octomap_cnn(
        model=model_instance,
        input_shape=(1, 18*36 + 8), # batch_size=1, feature_dim=648+8=656
        num_warmup=1000,  # 预热100次
        num_test=1000,   # 测试1000次
        use_profiler=False # 设置为 True 可以进行更详细的分析
    )

    # 4. 打印最终结果摘要
    print(f"\n[摘要] {model_instance.__class__.__name__} 在 GPU 上的平均推理时间为: {mean_time:.4f} ms")