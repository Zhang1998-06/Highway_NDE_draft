#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import single_lane_env as sle
from single_lane_env import CircularRaceTrackEnv
from utils import *
from policy import *

# ========= 评估配置 =========
EVAL_EPISODES      = 200
OBSERVED_COUNT     = 3
FRAMES_TO_PLAY     = 1
CIRCULAR_RADIUS    = 100
NUMBER_OF_VEHICLES = 5
MODEL_TAG          = "generate_data"

# checkpoint 默认路径（你也可以改成 "best"）
CKPT_DIR    = "checkpoints"
ACTOR_CKPT  = os.path.join(CKPT_DIR, "ppo_actor_final.pt")
CRITIC_CKPT = os.path.join(CKPT_DIR, "ppo_critic_final.pt")

# 画图参数
MEAN_NUMBER      = 10
SMOOTHING_WINDOW = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 把 FRAMES_TO_PLAY 注入 env 模块（initialize_environment/load_data 用到）
sle.FRAMES_TO_PLAY = FRAMES_TO_PLAY

def inject_targets_and_stats(env: CircularRaceTrackEnv,
                             circular_radius: float,
                             number_of_vehicle: int):
    targets = gt_distance_targets(strict=True)
    if targets is not None:
        D_MEAN, D_MIN, D_MAX = targets
    else:
        geom_mean = 4.0 * math.pi * circular_radius / number_of_vehicle
        D_MEAN, D_MIN, D_MAX = geom_mean, 0.6 * geom_mean, 1.8 * geom_mean

    GT_average_speed, _ = GT_macro()
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

def maybe_load_checkpoints(env: CircularRaceTrackEnv,
                           actor_ckpt: str,
                           critic_ckpt: str):
    loaded_any = False
    if actor_ckpt and os.path.exists(actor_ckpt):
        sd = torch.load(actor_ckpt, map_location=device)
        env.agent.actor.load_state_dict(sd)
        env.agent.actor.eval()
        print(f"[Eval] Loaded actor from {actor_ckpt}")
        loaded_any = True
    else:
        print(f"[Eval] Actor checkpoint not found: {actor_ckpt}")

    if critic_ckpt and os.path.exists(critic_ckpt):
        sd = torch.load(critic_ckpt, map_location=device)
        env.agent.critic.load_state_dict(sd)
        env.agent.critic.eval()
        print(f"[Eval] Loaded critic from {critic_ckpt}")
        loaded_any = True
    else:
        print(f"[Eval] Critic checkpoint not found: {critic_ckpt}")

    if not loaded_any:
        print("[Eval] No checkpoints loaded, evaluating with current/random weights.")

def evaluate_policy(model_tag: str,
                    circular_radius: float,
                    number_of_vehicle: int,
                    observed_count: int,
                    episodes: int,
                    actor_ckpt: str,
                    critic_ckpt: str):
    env = CircularRaceTrackEnv(model=model_tag,
                               circular_radius=circular_radius,
                               number_of_vehicle=number_of_vehicle)
    env.n_observed = int(observed_count)
    inject_targets_and_stats(env, circular_radius, number_of_vehicle)
    maybe_load_checkpoints(env, actor_ckpt, critic_ckpt)

    min_frame = env.vehicle_data['Frame'].min()
    max_frame = env.vehicle_data['Frame'].max() - FRAMES_TO_PLAY - 1

    returns = []
    for ep in range(episodes):
        start_frame = random.randint(min_frame, max_frame)
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

        avg_ep_ret = ep_return / max(1, steps)
        returns.append(avg_ep_ret)
        print(f"[Eval {ep+1}/{episodes}] steps={steps} avg_align_reward={avg_ep_ret:.4f}")

    pd.DataFrame({"avg_align_reward": returns}).to_csv("policy_eval_returns.csv", index=False)
    print("[Eval] Saved policy_eval_returns.csv")
    return returns

def plot_policy_eval_returns(returns,
                             mean_number=MEAN_NUMBER,
                             smoothing_window=SMOOTHING_WINDOW,
                             out_png="policy_eval_returns.png"):
    series = pd.Series(returns).rolling(window=smoothing_window, min_periods=1).mean()
    if len(series) >= mean_number:
        usable = series.values[: (len(series) // mean_number) * mean_number]
        mean_curve = np.mean(usable.reshape(-1, mean_number), axis=1)
        x_axis = np.arange(mean_number, mean_number * len(mean_curve) + 1, mean_number)
    else:
        mean_curve = series.values
        x_axis = np.arange(1, len(series) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, mean_curve, label="PPO (eval avg align reward)", linewidth=1)
    plt.xlabel("Episodes")
    plt.ylabel("Mean Align Reward")
    plt.title("Policy Evaluation: Mean Align Reward Over Episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[Eval] Saved {out_png}")

    mu = float(np.mean(returns)) if len(returns) > 0 else float("nan")
    sigma = float(np.std(returns)) if len(returns) > 0 else float("nan")
    print(f"[Eval] Avg Align Reward — mean: {mu:.4f}, std: {sigma:.4f}")
    return mu, sigma

if __name__ == "__main__":
    returns = evaluate_policy(
        model_tag=MODEL_TAG,
        circular_radius=CIRCULAR_RADIUS,
        number_of_vehicle=NUMBER_OF_VEHICLES,
        observed_count=OBSERVED_COUNT,
        episodes=EVAL_EPISODES,
        actor_ckpt=ACTOR_CKPT,
        critic_ckpt=CRITIC_CKPT
    )
    plot_policy_eval_returns(returns,
                             mean_number=MEAN_NUMBER,
                             smoothing_window=SMOOTHING_WINDOW,
                             out_png="policy_eval_returns.png")
