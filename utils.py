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
# Set random seeds for reproducibility

SEED = 42
MAX_SPEED = 14.0   

# Set device for PyTorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
import math
from itertools import permutations

def _arc_len(a, b, r):
    dtheta = abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    return 2 * dtheta * r

def match_generated_to_ids(generated_pairs, unobserved_ids, ref_states, r, w_theta=1.0, w_speed=0.0):
    """
    generated_pairs: list[(theta_gen, v_gen)]
    unobserved_ids: list[int]
    ref_states: {uid: (theta_ref, v_ref)}  # 从 env 中读出的真值
    r: 圆半径
    """
    k = len(generated_pairs)
    ids = list(unobserved_ids)

    # 代价矩阵
    C = [[0.0 for _ in range(k)] for _ in range(k)]
    for i, (theta_g, v_g) in enumerate(generated_pairs):
        for j, uid in enumerate(ids):
            theta_ref, v_ref = ref_states[uid]
            cost_theta = _arc_len(theta_g, theta_ref, r)
            cost_speed = abs(v_g - v_ref)
            C[i][j] = w_theta * cost_theta + w_speed * cost_speed

    # 小 k 直接全排列
    best_cost, best_perm = float('inf'), None
    for perm in permutations(range(k)):   # perm[i] = 生成索引，i 是 ID 列索引
        cost = sum(C[perm[i]][i] for i in range(k))
        if cost < best_cost:
            best_cost, best_perm = cost, perm

    return {ids[j]: generated_pairs[best_perm[j]] for j in range(k)}  # id -> (theta, v
def normalize_input(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    input_tensor: [1, 2N] = [θ0, v0, θ1, v1, ...]
    Returns a NEW normalized tensor; no in-place writes on views.
    """
    x = input_tensor.view(-1, 2)            # [N,2] view
    x = x.clone()                           # break view aliasing to avoid in-place-on-view issues

    theta = torch.remainder(x[:, 0], 2 * math.pi) / (2 * math.pi)
    speed = torch.clamp(x[:, 1], 0.0, MAX_SPEED) / MAX_SPEED

    out = torch.stack((theta, speed), dim=1).reshape(1, -1)   # NEW tensor
    out = out + torch.randn_like(out) * 1e-3                  # not in-place
    return out

"""def normalize_input(accepted_states,observed_states_tensor):
    generator_input= torch.tensor(accepted_states, dtype=torch.float32,
                                                device=device).unsqueeze(0)
    generator_input = (generator_input
                                + torch.randn_like(observed_states_tensor) * 0.1)"""


def load_vehicle_data(vehicle_data_path):
    return pd.read_csv(vehicle_data_path)

def load_speed_limits(speed_limits_path):
    return pd.read_csv(speed_limits_path)
def discretize(value, bins):
    """Discretize a continuous value into a given set of bins."""
    bin_idx = np.digitize([value], bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
    return bins[bin_idx[0]]

def discretize_array(values, bins):
    return np.array([discretize(v, bins) for v in values])

    

def load_distribution_from_csv():
    file_path = "data/highd/sim/generatorstatictic/distribution.csv"
    
    speed_distribution = {}
    distance_distribution = {}
    
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            bin_type = row[0]  # First column indicates whether it's "Speed" or "Distance"
            bin_str = row[1]   # Second column is the bin range as a string
            probability = float(row[2])  # Third column is the probability
            
            # Parse the bin range string, e.g., "[10.0,10.1)"
            bin_value = float(bin_str.strip('[]()').split(',')[0])
            
            # Store the probability in the appropriate dictionary based on bin type
            if bin_type == 'Speed':
                speed_distribution[bin_value] = probability
            elif bin_type == 'Distance':
                distance_distribution[bin_value] = probability

    return speed_distribution, distance_distribution

def GT_macro():
    speed_distribution, distance_distribution = load_distribution_from_csv()
    avg_speed = sum(speed * prob for speed, prob in speed_distribution.items())
    avg_distance = sum(distance * prob for distance, prob in distance_distribution.items())
    return avg_speed, avg_distance

def load_bins_from_distribution():
    speed_distribution, distance_distribution = load_distribution_from_csv()
    theta_bins = sorted(distance_distribution.keys())
    speed_bins = sorted(speed_distribution.keys())

    # Create bin-to-index mappings
    theta_bin_to_index = {bin_value: idx for idx, bin_value in enumerate(theta_bins)}
    speed_bin_to_index = {bin_value: idx for idx, bin_value in enumerate(speed_bins)}
    return theta_bins, speed_bins, theta_bin_to_index, speed_bin_to_index
def get_bin_index(value, bins, bin_to_index):
    # Find the closest bin for the value
    closest_bin = min(bins, key=lambda x: abs(x - value))
    return bin_to_index[closest_bin]
def load_gt_distance_hist_centered(path="data/highd/sim/generatorstatictic/distribution.csv"):
    """读取 Distance 直方图，并用 bin 的中心作为代表值。"""
    centers, probs = [], []
    with open(path, "r") as f:
        reader = csv.reader(f); next(reader)  # skip header
        for typ, bin_range, p in reader:
            if typ != "Distance":
                continue
            p = float(p)
            s = bin_range.strip()                   # 形如 "[150.0,200.0)"
            left  = float(s[s.find('[')+1 : s.find(',')])
            right = float(s[s.find(',')+1 : s.find(')')])
            center = 0.5 * (left + right)
            centers.append(center); probs.append(p)
    centers = np.array(centers, dtype=float)
    probs   = np.array(probs,   dtype=float)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    return centers, probs

def gt_distance_targets(path="data/highd/sim/generatorstatictic/distribution.csv",
                        strict=True, qmin=0.10, qmax=0.90):
    """
    Returns (D_MEAN, D_MIN, D_MAX) for Distance from distribution.csv.

    strict=True: use exact GT support [min_left_edge, max_right_edge] over bins with p>0.
    strict=False: use quantiles (qmin, qmax) for robustness to outliers / bin tails.
    """
    centers, probs = [], []
    left_edges, right_edges = [], []

    with open(path, "r") as f:
        reader = csv.reader(f); next(reader)
        for typ, bin_range, p in reader:
            if typ != "Distance":
                continue
            p = float(p)
            s = bin_range.strip()
            left  = float(s[s.find('[')+1 : s.find(',')])
            right = float(s[s.find(',')+1 : s.find(')')])
            center = 0.5*(left + right)

            left_edges.append(left)
            right_edges.append(right)
            centers.append(center)
            probs.append(p)

    centers = np.asarray(centers, dtype=float)
    probs   = np.asarray(probs,   dtype=float)
    left_edges  = np.asarray(left_edges,  dtype=float)
    right_edges = np.asarray(right_edges, dtype=float)

    if centers.size == 0 or probs.sum() == 0:
        return None  # caller will fall back to geometric values if needed

    probs = probs / probs.sum()

    # Mean from centers
    D_MEAN = float((centers * probs).sum())

    # Bounds
    mask = probs > 0
    if strict:
        D_MIN = float(left_edges[mask].min())
        D_MAX = float(right_edges[mask].max())
    else:
        # percentile bounds (robust option)
        order = np.argsort(centers)
        v = centers[order]; w = probs[order]
        cdf = np.cumsum(w); cdf = cdf / cdf[-1]
        D_MIN = float(np.interp(qmin, cdf, v))
        D_MAX = float(np.interp(qmax, cdf, v))

    return D_MEAN, D_MIN, D_MAX
def calculate_distance(x: torch.Tensor, y: torch.Tensor, circle_radius: float) -> torch.Tensor:
    """
    前向相邻弧长：从 x (后车) 到 y (前车) 的正向角差 Δθ ∈ [0, 2π)。
    注意：不要取“最短弧”(min(Δθ, 2π-Δθ))，那是错的。
    """
    delta = torch.remainder(y - x, 2 * math.pi)   # [0, 2π)
    return 2.0 * circle_radius * delta      




def save_trajectories(all_trajectories):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trajectory Index', 'Frame', 'Vehicle ID', 'Observed', 'Theta', 'Velocity', 'acc', 'Preceding', 'Following', 'v0', 'a_max', 'b', 'T', 's0', 's_alpha'])       
        for traj_index, trajectory in enumerate(all_trajectories):
            for frame, vehicles in trajectory.items():
                for vehicle_id, vehicle_data in vehicles.items():
                    relationships = vehicle_data.get('relationships', {})
                    idm_params = vehicle_data['idm_parameter']
                    row = [
                        traj_index,
                        frame,
                        vehicle_id,
                        vehicle_data.get('observed', ''),
                        vehicle_data.get('theta', ''),
                        vehicle_data.get('v', ''),
                        vehicle_data.get('acc', ""),
                        relationships.get('preceding', ''),
                        relationships.get('following', ''),
                        idm_params['v0'],
                        idm_params['a_max'],
                        idm_params['b'],
                        idm_params['T'],
                        idm_params['s0'],
                        idm_params['s_alpha']
                    ]
                    writer.writerow(row)

def plot_mean_rewards(ac_returns, mean_number, smoothing_window):
    ac_returns = [float(r) for r in ac_returns]

    ac_returns_series = pd.Series(ac_returns).rolling(window=smoothing_window, min_periods=1).mean()
    ac_mean_reward = np.mean(ac_returns_series.values.reshape(-1, mean_number), axis=1)
    mean_axis = np.arange(mean_number, len(ac_returns) + 1, mean_number)

    plt.figure(figsize=(10, 5))
    plt.plot(mean_axis, ac_mean_reward, label='PPO', linewidth=1)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title('Mean Returns Over Episodes')
    plt.legend()
    plt.show()

def safety_check(
    vehicle_list,
    candidate_theta,
    candidate_speed,          # kept for signature compatibility
    total_vehicle_number,
    needs,
    env,
    min_clearance=None        # NEW: optional override for clearance in meters
):
    """
    Returns True if the candidate is at least `min_clearance` away from
    every already-placed vehicle, measured along the ring. The check is
    symmetric (min of forward/backward arc).
    """
    # choose threshold
    thresh = float(min_clearance) if min_clearance is not None else float(SAFETY_DISTANCE)

    # iterate over *already filled* vehicles: indices of thetas are 0,2,4,...
    # we only consider slots that are already populated: 0 .. 2*(N - needs) - 1
    for i in range(0, 2 * (total_vehicle_number - needs), 2):
        theta_i = vehicle_list[i]

        # forward distance candidate -> existing
        d_fwd  = calculate_distance(candidate_theta, theta_i, env.circle_radius)
        # backward distance existing -> candidate
        d_back = calculate_distance(theta_i, candidate_theta, env.circle_radius)

        # use the smaller arc as the true separation
        # (detach to float for the comparison; we don't need gradients here)
        gap = float(torch.minimum(d_fwd, d_back).detach())

        if gap < thresh:
            return False  # early exit on first violation

    return True

def build_generator_input(accepted_states):
    """
    accepted_states: length=10 list of Tensors for 5 vehicles (2 each).
      e.g. [theta0, speed0, theta1, speed1, ..., theta4, speed4]
    Returns: A [1,10] Tensor for feeding into the generator.
    """
    # Stack them into a single 1D Tensor, then add batch_dim=1
    # Each entry in accepted_states is already a Tensor. We want to cat them along dim=0.
    input_tensor = torch.cat(accepted_states, dim=0).unsqueeze(0)  # shape [1,10]
    return input_tensor