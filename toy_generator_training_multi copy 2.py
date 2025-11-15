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
    generate_vehicle_number=1,
):
    # --- dims & models ---
    observed_dim = 2 * number_of_vehicle
    hidden_dim = 256

    GT_average_speed, GT_average_distance = GT_macro()
    state_generator = GeneratorModel(observed_dim, hidden_dim).to(device)
    env = CircularRaceTrackEnv(model=model, circular_radius=circular_radius, number_of_vehicle=number_of_vehicle)

    # --- distance targets from GT (strict) ---
    targets = gt_distance_targets(strict=True)
    if targets is not None:
        D_MEAN, D_MIN, D_MAX = targets
    else:
        geom_mean = 4.0 * math.pi * env.circle_radius / number_of_vehicle
        D_MEAN, D_MIN, D_MAX = geom_mean, 0.6 * geom_mean, 1.8 * geom_mean

    optimizer = optim.Adam(state_generator.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    min_frame = env.vehicle_data['Frame'].min()
    max_frame = env.vehicle_data['Frame'].max() - FRAMES_TO_PLAY - 1

    return_list = []

    # =============== TRAINING LOOP ===============
    for batch_index in range(int(TRAJECTORIES / batch_size)):
        optimizer.zero_grad()
        batch_loss = 0.0

        for _ in range(batch_size):
            # ---- context ----
            start_frame = random.choice(range(min_frame, max_frame))
            env.load_data(start_frame)

            unobserved_ids = random.sample(list(env.vehicles_in_sim[start_frame].keys()),
                                           generate_vehicle_number)
            observed_list = env.generator_obs(start_frame, unobserved_ids)

            accepted_states = [torch.tensor([val], dtype=torch.float32, device=device, requires_grad=False)
                               for val in observed_list]
            is_gen_flags = [False] * len(accepted_states)  # length = 2N

            needed = generate_vehicle_number
            fill_slot = 2 * (number_of_vehicle - needed)

            # ---- autoregressive fill ----
            while needed > 0:
                generator_input = build_generator_input(accepted_states)  # [1, 2N]
                generator_input = normalize_input(generator_input)

                cand_theta, cand_speed = state_generator(generator_input)
                cand_theta = cand_theta.view(-1)   # [1]
                cand_speed = cand_speed.view(-1)   # [1]

                accepted_states[fill_slot]   = cand_theta
                accepted_states[fill_slot+1] = cand_speed
                is_gen_flags[fill_slot] = True
                is_gen_flags[fill_slot+1] = True

                needed -= 1
                fill_slot = 2 * (number_of_vehicle - needed)

            # ---- pack & sort ----
            final_states_1d = torch.cat(accepted_states, dim=0)  # [2N]
            thetas = final_states_1d[0::2]                       # [N]
            speeds = final_states_1d[1::2]                       # [N]

            idx = torch.argsort(thetas)
            thetas_sorted = thetas[idx]
            speeds_sorted = speeds[idx]

            # ---- FIRST: compute ring distances ----
            N = thetas_sorted.shape[0]
            dists = [
                calculate_distance(thetas_sorted[i], thetas_sorted[(i + 1) % N], env.circle_radius)
                for i in range(N)
            ]
            dists_tensor = torch.stack(dists)  # [N]

            # ---- THEN: build mask for pairs touching generated cars ----
            is_gen_theta = torch.tensor(is_gen_flags[0::2], dtype=torch.bool, device=device)  # [N]
            is_gen_sorted = is_gen_theta[idx]
            pair_mask = is_gen_sorted | is_gen_sorted.roll(-1)  # [N]

            # ---- losses ----
            # speed mean alignment
            avg_speed = speeds_sorted.mean()
            loss_speed = F.mse_loss(avg_speed / GT_average_speed,
                                    torch.tensor(1.0, dtype=torch.float32, device=device))

            # distance loss on active edges only
            if pair_mask.any():
                d_active = dists_tensor[pair_mask]

                loss_dist_mean = F.mse_loss(d_active.mean() / D_MEAN,
                                            torch.tensor(1.0, dtype=torch.float32, device=device))

                vmin = F.relu(D_MIN - d_active) / D_MIN
                vmax = F.relu(d_active - D_MAX) / D_MAX
                loss_min_gap = (vmin ** 2).mean()
                loss_max_gap = (vmax ** 2).mean()

                dist_std = torch.std(d_active, unbiased=False)
                loss_dist_var = dist_std / (D_MEAN + 1e-6)
            else:
                loss_dist_mean = torch.tensor(0.0, device=device)
                loss_min_gap   = torch.tensor(0.0, device=device)
                loss_max_gap   = torch.tensor(0.0, device=device)
                loss_dist_var  = torch.tensor(0.0, device=device)

            W_MEAN, W_MIN, W_MAX, W_VAR = 0.5, 2.0, 2.0, 0.1
            loss_distance = (W_MEAN * loss_dist_mean
                           + W_MIN  * loss_min_gap
                           + W_MAX  * loss_max_gap
                           + W_VAR  * loss_dist_var)

            traj_loss = WEIGHT_SPEED * loss_speed + WEIGHT_DISTANCE * loss_distance
            batch_loss += traj_loss

        batch_loss /= batch_size
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(state_generator.parameters(), max_norm=1.0)
        optimizer.step()

        return_list.append(batch_loss.item())
        print(f"Batch {batch_index}, loss={batch_loss.item():.4f}")

    pd.DataFrame(return_list, columns=['loss']).to_csv(
        f'return_list_combined_generator_GEN{generate_vehicle_number}.csv',
        index=False
    )

    model_filename = (
        f'generator_combined_GEN{generate_vehicle_number}_'
        f'WEIGHTSPEED_{WEIGHT_SPEED:.2f}_WEIGHTDISTANCE_{WEIGHT_DISTANCE:.2f}.pth'
    )
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
        generate_vehicle_number=1
    )
    # Optionally plot results, etc.
    # plot_mean_rewards(mean_number, smoothing_window)
    # visualize_generator()
    # generate_and_collect_distribution(model="generate_data", circular_radius=100, number_of_vehicle=5)
    # plot_mean_distribution()
