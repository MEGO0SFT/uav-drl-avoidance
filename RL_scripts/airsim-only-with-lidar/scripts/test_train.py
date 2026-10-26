from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import gymnasium as gym

# 测试SubprocVecEnv和Monitor的基本用法
# 使用CartPole-v1环境进行测试

# 创建环境
def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)  # 使用Monitor包装环境以记录统计信息
        env.reset(seed=seed + rank)  # 使用新的API设置种子
        return env
    return _init
env_id = 'Pendulum-v1'
num_envs = 40

class LinearEntropyScheduler(BaseCallback):
    """
    线性调整熵系数的回调函数
    """
    def __init__(self, start_ent_coef, end_ent_coef, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.start_ent_coef = start_ent_coef
        self.end_ent_coef = end_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # 计算训练进度比例
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        # 线性计算当前的ent_coef
        current_ent_coef = self.start_ent_coef - (self.start_ent_coef - self.end_ent_coef) * progress
        
        # 更新模型优化器中的熵系数
        self.model.ent_coef = current_ent_coef
        # 记录当前值以便监控
        self.logger.record("train/ent_coef", current_ent_coef)
        return True

if __name__ == "__main__":
    entropy_scheduler = LinearEntropyScheduler(
    start_ent_coef=0.01,
    end_ent_coef=0.001,
    total_timesteps=700000)
    envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    policy_kwargs = dict(
            net_arch=dict(pi=[256, 128, 128], vf=[256, 128, 128])
        )
    # 创建PPO模型
    model = PPO("MlpPolicy", envs, verbose=1, batch_size=64, n_steps=128, gamma=0.99, learning_rate=2.5e-4, n_epochs=4, policy_kwargs=policy_kwargs, ent_coef=0.01)
    # 训练
    model.learn(total_timesteps=700000, callback=entropy_scheduler)
    # 评估模型,可视化评估过程
    envs.close()
    # 创建单独的渲染环境进行评估
    eval_env = gym.make(env_id, render_mode='human')
    eval_env = Monitor(eval_env)
    
    # 评估模型
    obs, _ = eval_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        eval_env.render()  # 这会显示可视化
        
        if terminated or truncated:
            obs, _ = eval_env.reset()
    
    eval_env.close()