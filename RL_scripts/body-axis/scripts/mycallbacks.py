import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from collections import deque

class ParallelTrainingStatsCallback(BaseCallback):
    """
    自定义回调函数，用于监控和记录智能体训练过程中的特定指标到TensorBoard中：
    - 超时率 (Timeout Rate)
    - 碰撞率 (Collision Rate)
    - 成功率 (Success Rate) - 基于最近 N 个 episodes
    - 进步率 (Progress Rate) - 基于 ep_rew_mean 的变化
    
    此版本支持并行环境。
    """

    def __init__(
        self,
        window_size: int = 500, # 用于计算成功率、超时率、碰撞率的滑动窗口大小
        log_interval: int = 500, # 每隔多少次 on_step 调用后进行一次日志记录
        verbose: int = 0,
        tensorboard_log: str = None, # TensorBoard 日志文件路径
    ):
        super(ParallelTrainingStatsCallback, self).__init__(verbose)
        self.window_size = window_size
        self.log_interval = log_interval
        self.tensorboard_log = tensorboard_log
        
        # 使用deque实现固定长度的滑动窗口
        self.success_deque = deque(maxlen=window_size)
        self.timeout_deque = deque(maxlen=window_size)
        self.collision_deque = deque(maxlen=window_size)

        # 上一次记录的平均奖励，用于计算进步率
        self.last_ep_rew_mean = None
        
        if tensorboard_log:
            self.tb_writer = TensorBoardOutputFormat(tensorboard_log)
        
    def _on_step(self) -> bool:
        """
        在每个环境步骤之后调用。
        收集单步信息。
        """
        # 获取 info 字典
        infos = self.locals.get('infos', [{}])
        dones = self.locals.get('dones', [False])

        for i, (info, done) in enumerate(zip(infos, dones)):
            if done:
                is_timeout = info.get('timeout', False)
                is_collision = info.get('collision_occurred', False)
                is_target_reach = info.get('target_reach', False)
                
                success = is_target_reach and not is_timeout and not is_collision
                
                self.success_deque.append(int(success))
                self.timeout_deque.append(int(is_timeout))
                self.collision_deque.append(int(is_collision))

        return True

    def _on_rollout_end(self) -> None:
        """
        在一次完整的 rollout 结束后调用。
        计算和记录统计数据到 TensorBoard。
        """
        num_eps = len(self.success_deque)

        if num_eps > 0:
            current_success_rate = sum(self.success_deque) / float(num_eps)
            current_timeout_rate = sum(self.timeout_deque) / float(num_eps)
            current_collision_rate = sum(self.collision_deque) / float(num_eps)
        else:
            current_success_rate = 0.0
            current_timeout_rate = 0.0
            current_collision_rate = 0.0

        # 获取平均奖励 ep_rew_mean
        ep_info_buffer = self.locals.get('ep_info_buffer')
        current_ep_rew_mean = np.mean([ep_info['r'] for ep_info in ep_info_buffer]) if ep_info_buffer else None

        current_progress_rate = 0.0
        if current_ep_rew_mean is not None and self.last_ep_rew_mean is not None:
            if self.last_ep_rew_mean != 0:
                current_progress_rate = (current_ep_rew_mean - self.last_ep_rew_mean) / abs(self.last_ep_rew_mean)
            elif current_ep_rew_mean > 0:
                current_progress_rate = float('inf')
            elif current_ep_rew_mean < 0:
                current_progress_rate = float('-inf')
            else:
                current_progress_rate = 0.0
        
        self.last_ep_rew_mean = current_ep_rew_mean

        total_timesteps = self.num_timesteps

        if self.tb_writer:
            tb_dict = {
                'train/success_rate': current_success_rate,
                'train/timeout_rate': current_timeout_rate,
                'train/collision_rate': current_collision_rate,
                'train/progress_rate': current_progress_rate if current_progress_rate is not None else 0.0,
            }
            try:
                # 兼容不同版本的写法
                if hasattr(self.tb_writer, 'write'):
                    self.tb_writer.write(tb_dict, key_excluded=None, step=total_timesteps)
                elif hasattr(self.tb_writer, 'add_scalar'):
                    for k, v in tb_dict.items():
                        self.tb_writer.add_scalar(k, v, total_timesteps)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Failed to write to TensorBoard: {e}")

        if self.verbose > 0 and (self.num_timesteps % self.log_interval == 0 or num_eps == 1):
            rew_mean_str = f"{current_ep_rew_mean:.2f}" if current_ep_rew_mean is not None else "N/A"
            prog_rate_str = f"{current_progress_rate*100:.2f}%" if current_progress_rate is not None else "N/A"
            print(f"Training Stats at step {total_timesteps}:")
            print(f"  Success Rate ({self.window_size} eps avg): {current_success_rate*100:.2f}%")
            print(f"  Timeout Rate ({self.window_size} eps avg): {current_timeout_rate*100:.2f}%")
            print(f"  Collision Rate ({self.window_size} eps avg): {current_collision_rate*100:.2f}%")
            print(f"  Avg Reward (ep_rew_mean): {rew_mean_str}")
            print(f"  Progress Rate: {prog_rate_str}")
            print("-" * 20)

    def _on_training_end(self) -> None:
        """
        训练结束后关闭 TensorBoard writer。
        """
        if self.tb_writer:
            self.tb_writer.close()