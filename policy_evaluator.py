#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ==== 本工程模块 ====
import single_lane_env as sle
from single_lane_env import CircularRaceTrackEnv
from utils import *                 # GT_macro, gt_distance_targets, calculate_distance, load_distribution_from_csv, load_gt_distance_hist_centered
from policy import *                # PPOContinuous, 等

# ========= 评估配置（按需改） =========
EVAL_FRAMES          = 400                # 采样多少个起始帧进行评估（每帧 rollout FRAMES_TO_PLAY 步）
OBSERVED_COUNT       = 3                  # 与训练一致；如想“全由策略控制”，可设为 0
FRAMES_TO_PLAY       = 1                  # 与训练一致
CIRCULAR_RADIUS      = 100
NUMBER_OF_VEHICLES   = 5
MODEL_TAG            = "generate_data"

# checkpoint（若无 final，自动回退到 best）
CKPT_DIR    = "checkpoints"
ACTOR_CKPT  = os.path.join(CKPT_DIR, "ppo_actor_final.pt")
CRITIC_CKPT = os.path.join(CKPT_DIR, "ppo_critic_final.pt")
if not os.path.exists(ACTOR_CKPT):
    ACTOR_CKPT = os.path.join(CKPT_DIR, "ppo_actor_best.pt")
if not os.path.exists(CRITIC_CKPT):
    CRITIC_CKPT = os.path.join(CKPT_DIR, "ppo_critic_best.pt")

# 直方图 bins（和你的生成器评估对齐）
SPEED_BINS     = np.arange(0, 31, 1)     # 0..30 m/s (左边界)
DISTANCE_BINS  = np.arange(0, 501, 25)   # 0..500 m（左边界）

# 设备/随机种子
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 把 FRAMES_TO_PLAY 注入到 env 模块（其 initialize_environment 会用）
sle.FRAMES_TO_PLAY = FRAMES_TO_PLAY


# ========= 与训练/生成器完全对齐的目标与权重注入 =========
def inject_targets_and_stats(env: CircularRaceTrackEnv,
                             circular_radius: float,
                             number_of_vehicle: int):
    targets = gt_distance_targets(strict=True)
    if targets is not None:
        D_MEAN, D_MIN, D_MAX = targets
    else:
        geom_mean = 4.0 * math.pi * circular_radius / number_of_vehicle
        D_MEAN, D_MIN, D_MAX = geom_mean, 0.6 * geom_mean, 1.8 * geom_mean

    GT_average_speed, _GT_avg_dist = GT_macro()
    S_MEAN = GT_average_speed

    # 距离项内部权重 —— 与生成器保持一致
    env.W_MEAN = 0.5
    env.W_MIN  = 1.0
    env.W_MAX  = 1.0
    env.W_VAR  = 0.1

    # 顶层融合权重 —— 与生成器一致
    env.WEIGHT_SPEED    = 0.7
    env.WEIGHT_DISTANCE = 1.0

    env.D_MEAN = float(D_MEAN)
    env.D_MIN  = float(D_MIN)
    env.D_MAX  = float(D_MAX)
    env.S_MEAN = float(S_MEAN)


def maybe_load_checkpoints(env: CircularRaceTrackEnv,
                           actor_ckpt: str,
                           critic_ckpt: str):
    loaded_any = False
    if actor_ckpt and os.path.exists(actor_ckpt):
        env.agent.actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
        env.agent.actor.eval()
        print(f"[Eval] Loaded actor from {actor_ckpt}")
        loaded_any = True
    else:
        print(f"[Eval] Actor checkpoint not found: {actor_ckpt}")

    if critic_ckpt and os.path.exists(critic_ckpt):
        env.agent.critic.load_state_dict(torch.load(critic_ckpt, map_location=device))
        env.agent.critic.eval()
        print(f"[Eval] Loaded critic from {critic_ckpt}")
        loaded_any = True
    else:
        print(f"[Eval] Critic checkpoint not found: {critic_ckpt}")

    if not loaded_any:
        print("[Eval] No checkpoints loaded; using current/random weights.")


# ========= 主评估：收集速度与“相邻弧长距离”，并做统计/画图 =========
def evaluate_policy_speed_distance(model_tag: str,
                                   circular_radius: float,
                                   number_of_vehicle: int,
                                   observed_count: int,
                                   sample_frames: int):
    env = CircularRaceTrackEnv(model=model_tag,
                               circular_radius=circular_radius,
                               number_of_vehicle=number_of_vehicle)
    env.n_observed = int(observed_count)
    inject_targets_and_stats(env, circular_radius, number_of_vehicle)
    maybe_load_checkpoints(env, ACTOR_CKPT, CRITIC_CKPT)

    # 可用帧范围
    min_frame = env.vehicle_data['Frame'].min()
    max_frame = env.vehicle_data['Frame'].max() - FRAMES_TO_PLAY - 1

    # 直方图计数器
    speed_counts    = {float(b): 0 for b in SPEED_BINS}
    distance_counts = {float(b): 0 for b in DISTANCE_BINS}

    # 原始取样（汇总统计用）
    all_speeds = []
    all_dists  = []

    for idx in range(sample_frames):
        start_frame = random.randint(min_frame, max_frame)
        env.initialize_environment(start_frame)

        done = False
        for _ in range(FRAMES_TO_PLAY):
            if done:
                break
            done, _align_r = env.step()

        # 取下一帧（当前 env.frame）所有车辆的速度/角度
        cur_frame = env.frame
        vehicles = env.vehicles_in_sim[cur_frame]

        # 收集速度
        speeds = [float(v['v']) for v in vehicles.values()]
        all_speeds.extend(speeds)

        # 按 theta 升序，计算“环上相邻弧长距离”（与你生成器评估一致）
        ordered = sorted(vehicles.values(), key=lambda x: x['theta'])
        n = len(ordered)
        r = env.circle_radius
        dists = []
        for i in range(n):
            theta_i = ordered[i]['theta']
            theta_j = ordered[(i + 1) % n]['theta']
            d = calculate_distance(torch.tensor(theta_i), torch.tensor(theta_j), r)
            dists.append(float(d))
        all_dists.extend(dists)

        # 累计到直方图（选最近的“左边界 bin”）
        for sp in speeds:
            sp_bin = min(speed_counts.keys(), key=lambda b: abs(b - sp))
            speed_counts[sp_bin] += 1
        for ds in dists:
            ds_bin = min(distance_counts.keys(), key=lambda b: abs(b - ds))
            distance_counts[ds_bin] += 1

        if (idx + 1) % max(1, sample_frames // 10) == 0:
            print(f"[Eval] {idx+1}/{sample_frames} frames processed")

    # 归一化为分布（概率）
    total_speed = sum(speed_counts.values())
    total_dist  = sum(distance_counts.values())
    speed_dist  = {b: (c / total_speed if total_speed > 0 else 0.0) for b, c in speed_counts.items()}
    dist_dist   = {b: (c / total_dist  if total_dist  > 0 else 0.0) for b, c in distance_counts.items()}

    # 保存分布 CSV（与生成器评估格式一致）
    with open("policy_distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Bin Type", "Bin", "Probability"])
        for b, p in speed_dist.items():
            w.writerow(["Speed", b, p])
        for b, p in dist_dist.items():
            w.writerow(["Distance", b, p])
    print("[Eval] Saved policy_distribution.csv")

    # 统计摘要
    def stats(x):
        x = np.array(x, dtype=float)
        if x.size == 0:
            return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, n=0)
        return dict(mean=float(np.mean(x)),
                    std=float(np.std(x)),
                    min=float(np.min(x)),
                    max=float(np.max(x)),
                    n=int(x.size))

    speed_stats = stats(all_speeds)
    dist_stats  = stats(all_dists)

    # 保存摘要 CSV
    pd.DataFrame([
        {"metric": "speed",    **speed_stats},
        {"metric": "distance", **dist_stats}
    ]).to_csv("policy_eval_stats.csv", index=False)
    print("[Eval] Saved policy_eval_stats.csv")

    # 在终端打印
    print(f"[Eval] Speed    — mean: {speed_stats['mean']:.4f}, std: {speed_stats['std']:.4f}, "
          f"min: {speed_stats['min']:.4f}, max: {speed_stats['max']:.4f}, n={speed_stats['n']}")
    print(f"[Eval] Distance — mean: {dist_stats['mean']:.4f}, std: {dist_stats['std']:.4f}, "
          f"min: {dist_stats['min']:.4f}, max: {dist_stats['max']:.4f}, n={dist_stats['n']}")

    return speed_dist, dist_dist, speed_stats, dist_stats


# ========= 画与 GT 的分布对比（沿用你的生成器评估风格） =========
def plot_policy_vs_gt():
    # GT：距离用中心值读取，速度用你现有 CSV 解析（左边界）
    centers, probs = load_gt_distance_hist_centered()
    dist_gt = {float(c): float(p) for c, p in zip(centers, probs)}
    speed_gt, _unused = load_distribution_from_csv()  # {left_edge: prob}

    # Policy：从刚保存的 CSV 读回
    pol_speed, pol_dist = {}, {}
    with open("policy_distribution.csv", "r") as f:
        reader = csv.reader(f); next(reader)
        for typ, b, p in reader:
            b, p = float(b), float(p)
            if typ == "Speed":
                pol_speed[b] = p
            elif typ == "Distance":
                pol_dist[b] = p

    # 对齐 bins（并集补零）
    all_sp_bins = sorted(set(speed_gt.keys()) | set(pol_speed.keys()))
    all_ds_bins = sorted(set(dist_gt.keys())  | set(pol_dist.keys()))
    speed_gt  = {b: speed_gt.get(b, 0.0)  for b in all_sp_bins}
    pol_speed = {b: pol_speed.get(b, 0.0) for b in all_sp_bins}
    dist_gt   = {b: dist_gt.get(b, 0.0)   for b in all_ds_bins}
    pol_dist  = {b: pol_dist.get(b, 0.0)  for b in all_ds_bins}

    # 速度分布对比
    plt.figure(figsize=(10, 5))
    plt.bar(list(speed_gt.keys()), list(speed_gt.values()), alpha=0.5, label="GT Speed", width=0.4)
    plt.bar(list(pol_speed.keys()), list(pol_speed.values()), alpha=0.5, label="Policy Speed", width=0.2)
    plt.xticks(rotation=90)
    plt.title("Speed Distribution: Policy vs GT")
    plt.xlabel("Speed Bin (m/s, left edge)")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("policy_speed_distribution_comparison.png")
    plt.close()
    print("[Eval] Saved policy_speed_distribution_comparison.png")

    # 距离分布对比
    plt.figure(figsize=(10, 5))
    plt.bar(list(dist_gt.keys()), list(dist_gt.values()), alpha=0.5, label="GT Distance", width=15)
    plt.bar(list(pol_dist.keys()), list(pol_dist.values()), alpha=0.5, label="Policy Distance", width=7)
    plt.xticks(rotation=90)
    plt.title("Adjacent Distance Distribution: Policy vs GT")
    plt.xlabel("Distance Bin (m, center or left edge)")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("policy_distance_distribution_comparison.png")
    plt.close()
    print("[Eval] Saved policy_distance_distribution_comparison.png")

    # 打印 GT 的均值（供参考）
    def expected_from_hist(d):
        s = sum(d.values())
        return float("nan") if s <= 0 else sum(b * p for b, p in d.items()) / s
    print(f"[GT] Avg speed (from GT csv): {expected_from_hist(speed_gt):.3f}")
    print(f"[GT] Avg adjacent distance  : {expected_from_hist(dist_gt):.3f}")


# ========= 主入口 =========
if __name__ == "__main__":
    speed_dist, dist_dist, speed_stats, dist_stats = evaluate_policy_speed_distance(
        model_tag=MODEL_TAG,
        circular_radius=CIRCULAR_RADIUS,
        number_of_vehicle=NUMBER_OF_VEHICLES,
        observed_count=OBSERVED_COUNT,     # 如想“全策略控制”，可改为 0
        sample_frames=EVAL_FRAMES
    )
    plot_policy_vs_gt()
