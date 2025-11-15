import pandas as pd
import numpy as np
import csv
import random
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from single_lane_env import CircularRaceTrackEnv
from utils import *
from Generator_model import GeneratorModel




SEED = 42
DROP_OUT_P = 0.5
TRAJECTORIES =4000
FRAMES_TO_PLAY = 1
GENERATOR_BATCH=20
SEGMENT_LEN = 100
# the number of update is trajectories/batch*framestoPLAY
# Hyperparameters
LEARNING_RATE = 1e-4
# should not be larger than1e-4 0.762 
# the loss is 0.14, 
GAMMA = 0.98
LAMBDA = 0.95
EPOCHS = 10

MAX_THETA = 2*math.pi      # e.g. allow theta in [-pi, pi] radians (adjust as needed)
MAX_SPEED = 14.0         # e.g. maximum speed in m/s (adjust as needed)
WEIGHT_SPEED = 0.7
WEIGHT_DISTANCE = 1
SAFETY_DISTANCE = 100
SAFETY_DISTANCE_PENALTY = 1  # not used directly below
# Environment and data paths
segmentPath = f"toy_study/data/_{SEGMENT_LEN}_{SEED}_{DROP_OUT_P}"
vehicle_data_path = f"data/highd/sim/_{SEED}_{DROP_OUT_P}.csv"
speed_limits_path = vehicle_data_path.replace(".csv", "_speed_limits.csv")
output_path = "data/highd/sim/combined_trajectories_generator.csv"


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
# never,never add hard constraint during training !!!!! never!!!!
# -----------------------------------------------------------------------------
# [CHANGED] Helper function to build generator input from accepted_states
# -----------------------------------------------------------------------------

def collect_combined_distribution_with_generator(
    model,
    circular_radius,
    number_of_vehicle,
    batch_size=GENERATOR_BATCH,
    generate_vehicle_number=2
):
    # ---------------------------------------------------------
    # CHANGED: 动态决定输入维度（每车2维）
    # ---------------------------------------------------------
    observed_dim = 2 * number_of_vehicle
    hidden_dim = 256

    theta_bins, speed_bins, theta_bin_to_index, speed_bin_to_index = load_bins_from_distribution()
    GT_average_speed, GT_average_distance = GT_macro()

    state_generator = GeneratorModel(observed_dim, hidden_dim).to(device)
    env = CircularRaceTrackEnv(model=model, circular_radius=circular_radius, number_of_vehicle=number_of_vehicle)
    targets = gt_distance_targets(strict=True) 
    if targets is not None:
        D_MEAN, D_MIN, D_MAX = targets
    else:
        geom_mean = 4.0 * math.pi * env.circle_radius / number_of_vehicle  # consistent with 2*R*Δθ
        D_MEAN = geom_mean
        D_MIN  = 0.6 * geom_mean
        D_MAX  = 1.8 * geom_mean
    print(f"[Targets] D_MEAN={D_MEAN:.2f}, D_MIN={D_MIN:.2f}, D_MAX={D_MAX:.2f}")


    speed_distribution, distance_distribution = load_distribution_from_csv()
    optimizer = optim.Adam(state_generator.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    min_frame = env.vehicle_data['Frame'].min()
    max_frame = env.vehicle_data['Frame'].max() - FRAMES_TO_PLAY - 1

    return_list = []

    for batch_index in range(int(TRAJECTORIES / batch_size)):
        optimizer.zero_grad()
        batch_loss = 0.0

        for trajectory_index in range(batch_size):
            start_frame = random.choice(range(min_frame, max_frame))
            env.load_data(start_frame)

            unobserved_ids = random.sample(
                list(env.vehicles_in_sim[start_frame].keys()),
                generate_vehicle_number
            )
            observed_list = env.generator_obs(start_frame, unobserved_ids)

            # 观测是常数，不需要梯度
            accepted_states = [torch.tensor([val], dtype=torch.float32, device=device, requires_grad=False)
                               for val in observed_list]

            needed = generate_vehicle_number
            fill_slot = 2 * (number_of_vehicle - needed)

            while needed > 0:
                generator_input = build_generator_input(accepted_states)     # [1, 2N]
                generator_input = normalize_input(generator_input)

                candidate_theta, candidate_speed = state_generator(generator_input)
                candidate_theta = candidate_theta.view(-1)  # [1]
                candidate_speed = candidate_speed.view(-1)  # [1]

                # 直接写入占位，保持梯度链
                accepted_states[fill_slot]   = candidate_theta
                accepted_states[fill_slot+1] = candidate_speed

                needed -= 1
                fill_slot = 2 * (number_of_vehicle - needed)

            # 拼回整帧状态（保持梯度）
            final_states_1d = torch.cat(accepted_states, dim=0)   # [2N]
            thetas = final_states_1d[0::2]                        # [N]
            speeds = final_states_1d[1::2]                        # [N]

            # ---------------------------------------------------------
            # CHANGED: 按角度排序再计算“环上相邻”——保证物理邻接
            # ---------------------------------------------------------
            idx = torch.argsort(thetas)
            thetas_sorted = thetas[idx]
            speeds_sorted = speeds[idx]

            # 速度损失（与 GT 均速对齐）
            avg_speed = speeds_sorted.mean()
            loss_speed = F.mse_loss(
                avg_speed / GT_average_speed,
                torch.tensor(1.0, dtype=torch.float32, device=device)
            )

            # 相邻弧长距离（张量版，保留梯度）
            dists = []
            N = thetas_sorted.shape[0]
            for i in range(N):
                j = (i + 1) % N
                dists.append(calculate_distance(thetas_sorted[i], thetas_sorted[j], env.circle_radius))
            dists_tensor = torch.stack(dists)          # [N]
            avg_distance = dists_tensor.mean()         # 平均相邻距离

            W_MEAN = 0.5          # 均值约束权重
            W_MIN  = 1.0           # 最小间距铰链权重
            W_MAX  = 1.0          # 最大间距铰链权重（通常比下界小）
            W_VAR  = 0.1           # （可选）方差正则，鼓励不过分离散

            # 1) 均值约束（不需要 GT 曲线，只对齐给定平均值）
            loss_dist_mean = F.mse_loss(avg_distance / D_MEAN,
                                        torch.tensor(1.0, dtype=torch.float32, device=device))

            # 2) 最小/最大间距软约束（铰链）：只惩罚违反者
            violate_min = F.relu(D_MIN - dists_tensor) / D_MIN   # < D_MIN 才产生损失
            violate_max = F.relu(dists_tensor - D_MAX) / D_MAX   # > D_MAX 才产生损失
            loss_min_gap = violate_min.mean()
            loss_max_gap = violate_max.mean()

            # 3) （可选）幅度正则：控制离散度，避免“有的太小、有的太大”
            dist_std = torch.std(dists_tensor, unbiased=False)
            # 可用“相对标准差”归一化，免得半径变更导致尺度跳动
            loss_dist_var = (dist_std / (D_MEAN + 1e-6))




            # 最终距离项（只基于 min/max/mean，不用完整 GT 分布）
            loss_distance = (W_MEAN * loss_dist_mean
                            + W_MIN  * loss_min_gap
                            + W_MAX  * loss_max_gap
                            + W_VAR  * loss_dist_var)/(W_MEAN+W_MIN + W_MAX + W_VAR )

            # 合并总损失
            traj_loss = WEIGHT_SPEED * loss_speed + WEIGHT_DISTANCE * loss_distance
            batch_loss += traj_loss

        batch_loss /= batch_size
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(state_generator.parameters(), max_norm=1.0)
        optimizer.step()

        return_list.append(batch_loss.item())
        print(f"Batch {batch_index}, loss={batch_loss.item():.4f}")

    pd.DataFrame(return_list, columns=['loss']).to_csv('return_list_combined_generator.csv', index=False)
    model_filename = f'generator_combined_WEIGHTSPEED_{WEIGHT_SPEED:.2f}_WEIGHTDISTANCE_{WEIGHT_DISTANCE:.2f}.pth'
    torch.save(state_generator.state_dict(), model_filename)
    print(f"Model saved as: {model_filename}")

    return return_list


if __name__ == "__main__":
    mean_number = 10
    smoothing_window = 10
    return_list = collect_combined_distribution_with_generator(
        model="generate_data",
        circular_radius=100,
        number_of_vehicle=5,
        generate_vehicle_number=4
    )
    # Optionally plot results, etc.
    # plot_mean_rewards(mean_number, smoothing_window)
    # visualize_generator()
    # generate_and_collect_distribution(model="generate_data", circular_radius=100, number_of_vehicle=5)
    # plot_mean_distribution()
