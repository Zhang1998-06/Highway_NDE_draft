# policy_training.py
import os                       # ← 新增
import math
import copy
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from policy import *
from utils import *  # 需要: gt_distance_targets, GT_macro, save_trajectories, plot_mean_rewards 等
from single_lane_env import CircularRaceTrackEnv

# ---------------- 新增：保存相关配置 ----------------
OUT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)
# 每多少个 episode 存一次快照；也会保存 best 和 final
SAVE_EVERY =  max(1, TRAJECTORIES // 10) if "TRAJECTORIES" in globals() else 100

def save_agent_ckpt(agent, out_dir, tag="final"):
    """保存 PPO 策略参数（actor/critic 各一份），以及一个打包文件。"""
    actor_path  = os.path.join(out_dir, f"ppo_actor_{tag}.pt")
    critic_path = os.path.join(out_dir, f"ppo_critic_{tag}.pt")
    both_path   = os.path.join(out_dir, f"ppo_{tag}.pt")

    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    torch.save({"actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict()}, both_path)
    print(f"[Save] actor -> {actor_path}")
    print(f"[Save] critic -> {critic_path}")
    print(f"[Save] both   -> {both_path}")

# 数据路径（与你现有工程一致）
segmentPath = f"toy_study/data/_{SEGMENT_LEN}_{SEED}_{DROP_OUT_P}"
vehicle_data_path = f"data/highd/sim/_{SEED}_{DROP_OUT_P}.csv"
speed_limits_path = vehicle_data_path.replace(".csv", "_speed_limits.csv")
output_path = "data/highd/sim/combined_trajectories.csv"

# 设备与随机种子
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -------------------
# 训练专属的高层配置（集中管理）
# -------------------
OBSERVED_COUNT = 3

# policy_training.py
def inject_targets_and_stats(env, circular_radius, number_of_vehicle):
    # 与生成器对齐（保持你给我的设置不变）
    targets = gt_distance_targets(strict=True)
    if targets is not None:
        D_MEAN, D_MIN, D_MAX = targets
    else:
        geom_mean = 4.0 * math.pi * circular_radius / number_of_vehicle
        D_MEAN, D_MIN, D_MAX = geom_mean, 0.6 * geom_mean, 1.8 * geom_mean

    GT_average_speed, GT_average_distance = GT_macro()
    S_MEAN = GT_average_speed

    env.W_MEAN = 0.5
    env.W_MIN  = 1.0
    env.W_MAX  = 1.0
    env.W_VAR  = 0.1

    env.WEIGHT_SPEED    = 0.7
    env.WEIGHT_DISTANCE = 1.0

    env.D_MEAN = float(D_MEAN)
    env.D_MIN  = float(D_MIN)
    env.D_MAX  = float(D_MAX)
    env.S_MEAN = float(S_MEAN)


def collect_data_with_ppo(model, circular_radius, number_of_vehicle, observed_count=OBSERVED_COUNT):
    """
    训练逻辑（保持不变）：
    - 每步 PPO 为 5 辆车输出动作；观测车执行 GT，不观测车执行 PPO；
    - 计算全体车辆的速度/距离对齐奖励；episode 末统一推入回放池；
    - 做一批 PPO 更新。
    """
    env = CircularRaceTrackEnv(model=model,
                               circular_radius=circular_radius,
                               number_of_vehicle=number_of_vehicle)

    env.n_observed = int(observed_count)
    inject_targets_and_stats(env, circular_radius, number_of_vehicle)

    return_list = []
    best_running = -1e9  # 跟踪最佳平均回报（简单用最后一步的 avg 作为度量）
    running_mean_window = 20
    recent = []

    for ep in range(int(TRAJECTORIES)):
        # 采样起始帧
        min_frame = env.vehicle_data['Frame'].min()
        max_frame = env.vehicle_data['Frame'].max() - FRAMES_TO_PLAY - 1
        start_frame = random.randint(min_frame, max_frame)

        # 初始化 episode
        env.initialize_environment(start_frame)

        ep_return = 0.0
        steps = 0
        done = False

        for _ in range(FRAMES_TO_PLAY):
            if done:
                break
            done, align_r = env.step()
            ep_return += float(align_r)
            steps += 1

        # 统一推入回放池 + 更新
        env.flush_episode_memory()

        num_minibatches = max(1, len(env.replay_memory) // BATCH_SIZE)
        for _ in range(num_minibatches):
            env.agent.update_controller(env.replay_memory)

        avg_ep_return = ep_return / max(1, steps)
        return_list.append(avg_ep_return)
        recent.append(avg_ep_return)
        if len(recent) > running_mean_window:
            recent.pop(0)
        running_mean = float(np.mean(recent))

        print(f"[Episode {ep+1}/{TRAJECTORIES}] steps={steps} avg_align_reward={avg_ep_return:.4f} "
              f"running_mean({len(recent)}ep)={running_mean:.4f} replay_size={len(env.replay_memory)}")

        # -------- 新增：按间隔保存快照 --------
        if (ep + 1) % SAVE_EVERY == 0:
            save_agent_ckpt(env.agent, OUT_DIR, tag=f"ep{ep+1}")

        # -------- 新增：保存最佳（按 running_mean 度量） --------
        if running_mean > best_running:
            best_running = running_mean
            save_agent_ckpt(env.agent, OUT_DIR, tag="best")

    # 训练曲线
    pd.DataFrame(return_list, columns=['return']).to_csv('return_list_agent.csv', index=False)
    plot_mean_rewards(return_list, mean_number=10, smoothing_window=10)

    # -------- 新增：保存最终模型 --------
    save_agent_ckpt(env.agent, OUT_DIR, tag="final")


if __name__ == "__main__":
    collect_data_with_ppo(model="generate_data",
                          circular_radius=100,
                          number_of_vehicle=5,
                          observed_count=OBSERVED_COUNT)
    ac_returns = pd.read_csv('return_list_agent.csv').squeeze().tolist()
    plot_mean_rewards(ac_returns, mean_number=10, smoothing_window=10)
