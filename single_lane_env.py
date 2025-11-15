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
from policy import *
from utils import *



# Environment and data paths
segmentPath = f"toy_study/data/_{SEGMENT_LEN}_{SEED}_{DROP_OUT_P}"
vehicle_data_path = f"data/highd/sim/_{SEED}_{DROP_OUT_P}.csv"
speed_limits_path = vehicle_data_path.replace(".csv", "_speed_limits.csv")
output_path = "data/highd/sim/combined_trajectories_generator.csv"

# Set device for PyTorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)



    
class CircularRaceTrackEnv:
    def __init__(self, model, circular_radius=10, number_of_vehicle=10,observed_count=None):
        self.action_space = 1  # continuous action space
        self.n_vehicles = number_of_vehicle
        self.number_of_vehicle = number_of_vehicle
        self.observation_space = 4  

        self.model = model
        self.vehicles_in_sim = {}
        self.simulation_duration = 10000
        self.Speed_reward_weight = 1
        self.circle_radius = circular_radius
        self.frame = 0
        self.vehicle_data = load_vehicle_data(vehicle_data_path)
        self.speed_limits = load_speed_limits(speed_limits_path)
        self.n_observed = observed_count if observed_count is not None else number_of_vehicle
        self.agent = PPOContinuous(state_dim=self.observation_space, hidden_dim=HIDDEN_DIM, action_dim=self.action_space,
                         actor_lr=LEARNING_RATE_ACTOR, critic_lr=LEARNING_RATE_CRITIC, lmbda=LAMBDA, epochs=EPOCHS,
                         eps=EPS_CLIP, gamma=GAMMA, device=device, batch_size=BATCH_SIZE)
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    def reset(self):
        self.frame = 0
        self.vehicles_in_sim = {}

    def get_speed_limit(self, theta, frame):
        # Guard against rare holes by lazily caching the frame's zones
        zones = self.speed_zones_per_frame.get(frame)
        if zones is None:
            zdf = self.speed_limits[self.speed_limits['Frame'] == frame]
            if zdf.empty:
                raise KeyError(f"Speed-limit zones missing for frame {frame}")
            zones = [
                (float(r['Start Angle']), float(r['End Angle']), float(r['Speed Limit']))
                for _, r in zdf.iterrows()
            ]
            self.speed_zones_per_frame[frame] = zones

        for start_angle, end_angle, speed_limit in zones:
             if start_angle < end_angle:
                 if start_angle <= theta < end_angle:
                     return speed_limit
             else:  # Wrap around case
                 if theta >= start_angle or theta < end_angle:
                     return speed_limit
        # Numerical safety: if no interval matched (e.g., boundary rounding), return the first zone's limit
        return zones[0][2]

    def crashes(self, vehicle_state, next_frame, veh_id):
        crash = False
        theta, cl = vehicle_state["theta"], 4.5
        for other_veh_id, other_vehicle_state in self.vehicles_in_sim[next_frame].items():
            if other_veh_id != veh_id:
                other_theta, other_l = other_vehicle_state["theta"], 4.5
                if abs(theta - other_theta) <= (cl + other_l) / (2 * self.circle_radius):
                    crash = True
        return crash
    def execute_action(self, current_velocity, action, speed_limit):
        # Assume action is continuous and maps to acceleration
        acceleration = action[0] if isinstance(action, np.ndarray) else action 
        new_velocity = np.clip(current_velocity + acceleration * 0.04, 0, speed_limit)
        return new_velocity

    def obs(self, frame, veh_id, speed_limit):
        def check_ROI():
            def get_relationship_distance(category):
                relationship = self.vehicles_in_sim[frame][veh_id]['relationships'].get(category)
                if relationship is not None and relationship[1] is not None:
                    distance = relationship[1]
                    if category == 'preceding':
                        return distance
                    elif category == 'following':
                        return distance
                else:
                    return self.circle_radius * 2  # Maximum possible distance
            return (get_relationship_distance('preceding'), get_relationship_distance('following'))

        preceding_distance, _ = check_ROI()
        ego_velocity = self.vehicles_in_sim[frame][veh_id]['v']
        preceding_vehicle_id = self.vehicles_in_sim[frame][veh_id]['relationships'].get('preceding')[0] if self.vehicles_in_sim[frame][veh_id]['relationships'].get('preceding') else None
        preceding_velocity = self.vehicles_in_sim[frame][preceding_vehicle_id]['v'] if preceding_vehicle_id is not None else 0
        relative_speed_with_preceding = ego_velocity - preceding_velocity

        state = [
            ego_velocity,
            speed_limit,
            preceding_distance,
            relative_speed_with_preceding
        ]

        return state
    # single_lane_env.py  (class CircularRaceTrackEnv 内)
    def step(self):
        """
        每步为所有车计算动作：
        - 观测车：执行 GT 动作；不观测车：执行 RL 动作
        - 用“与生成器一致”的速度/距离损失构造对齐奖励 align_reward ∈ [0,1]
        - 只暂存 transition，不在此处推入回放池；在 flush_episode_memory() 统一推入
        """
        frame = self.frame
        next_frame = frame + 1

        # 确保下一帧容器存在
        self.vehicles_in_sim[next_frame] = {}

        # 1) 采样 RL 动作（观测车稍后会被 GT 覆盖）
        act_cache = {}
        for vid, vcur in self.vehicles_in_sim[frame].items():
            sl = self.get_speed_limit(vcur['theta'], frame)
            s  = self.obs(frame, vid, sl)
            a_rl, logp_rl = self.agent.take_action(s)
            a_rl = float(np.atleast_1d(a_rl)[0])
            act_cache[vid] = (a_rl, float(logp_rl), s, sl)

        # 2) 执行动作：观测车用 GT，加速度写回；不观测车用 RL
        for vid, vcur in self.vehicles_in_sim[frame].items():
            a_rl, logp_rl, s, sl = act_cache[vid]
            if vcur.get('observed', True):
                a_exec = float(vcur['acc'])  # GT
            else:
                a_exec = a_rl                # RL 控制
                vcur['acc'] = a_exec
            new_v = self.execute_action(vcur['v'], a_exec, sl)
            self.position_update_PPO_control(next_frame, vid, vcur, new_v)

        # 3) 更新下一帧前后关系
        self.update_surrounding(next_frame)

        # 4) 与生成器一致的损失 -> 对齐奖励
        if not all(hasattr(self, k) for k in ['D_MEAN', 'D_MIN', 'D_MAX', 'S_MEAN']):
            raise RuntimeError("缺少 D_/S_ 统计，请先调用 inject_targets_and_stats() 注入环境参数。")

        # 收集速度与“前车距离”（与生成器相同口径：正向弧长）
        speeds, dists = [], []
        for vid, vnext in self.vehicles_in_sim[next_frame].items():
            speeds.append(vnext['v'])
            pre = vnext['relationships'].get('preceding')
            dists.append(pre[1] if (pre is not None and pre[1] is not None) else self.circle_radius * 2)

        speeds = torch.as_tensor(speeds, dtype=torch.float, device=device)
        dists  = torch.as_tensor(dists,  dtype=torch.float, device=device)

        # --- 距离项：完全复刻生成器 ---
        if dists.numel() > 0:
            # 平均距离对齐到 D_MEAN
            loss_dist_mean = F.mse_loss(dists.mean() / (self.D_MEAN + 1e-6),
                                        torch.tensor(1.0, dtype=torch.float, device=device))
            # 软约束：只惩罚越界（不平方，与生成器一致）
            violate_min = F.relu(self.D_MIN - dists) / (self.D_MIN + 1e-6)
            violate_max = F.relu(dists - self.D_MAX) / (self.D_MAX + 1e-6)
            loss_min_gap = violate_min.mean()
            loss_max_gap = violate_max.mean()
            # 方差正则：相对标准差
            dist_std = torch.std(dists, unbiased=False)
            loss_dist_var = (dist_std / (self.D_MEAN + 1e-6))
        else:
            loss_dist_mean = loss_min_gap = loss_max_gap = loss_dist_var = torch.tensor(0.0, device=device)

        W_MEAN = getattr(self, 'W_MEAN', 0.5)
        W_MIN  = getattr(self, 'W_MIN',  1.0)
        W_MAX  = getattr(self, 'W_MAX',  1.0)
        W_VAR  = getattr(self, 'W_VAR',  0.1)
        _wsum  = (W_MEAN + W_MIN + W_MAX + W_VAR) + 1e-12

        loss_distance = (W_MEAN * loss_dist_mean
                        + W_MIN  * loss_min_gap
                        + W_MAX  * loss_max_gap
                        + W_VAR  * loss_dist_var) / _wsum

        # --- 速度项：完全复刻生成器（仅对齐平均速度到 S_MEAN；不加限速惩罚）---
        if speeds.numel() > 0:
            loss_speed = F.mse_loss(speeds.mean() / (self.S_MEAN + 1e-6),
                                    torch.tensor(1.0, dtype=torch.float, device=device))
        else:
            loss_speed = torch.tensor(0.0, device=device)

        WEIGHT_SPEED    = getattr(self, 'WEIGHT_SPEED', 0.7)
        WEIGHT_DISTANCE = getattr(self, 'WEIGHT_DISTANCE', 1.0)
        traj_loss = WEIGHT_SPEED * loss_speed + WEIGHT_DISTANCE * loss_distance

        # 将损失映射为 [0,1] 的对齐奖励
        align_reward = float(torch.clamp(1.0 / (1.0 + traj_loss), 0.0, 1.0).item())
        self._episode_align_sum += align_reward
        self._episode_steps += 1

        # 5) 暂存 transition（不立刻 push）
        done_any = False
        for vid, vcur in self.vehicles_in_sim[frame].items():
            cur_sl = self.get_speed_limit(vcur['theta'], frame)
            cur_s  = self.obs(frame, vid, cur_sl)
            vnext  = self.vehicles_in_sim[next_frame][vid]
            nxt_sl = self.get_speed_limit(vnext['theta'], next_frame)
            nxt_s  = self.obs(next_frame, vid, nxt_sl)

            d = self.crashes(vnext, next_frame, vid)
            done_any = done_any or d

            a_rl, logp_rl, _, _ = act_cache[vid]
            if vcur.get('observed', True):
                # 计算 GT 动作在当前策略下的 logp（监督到策略）
                st = torch.tensor([cur_s], dtype=torch.float32, device=device)
                mu, std = self.agent.actor(st)
                mu = torch.clamp(mu, -1.1, 0.5)
                dist = torch.distributions.Normal(mu, std)
                a_gt_t = torch.tensor([[float(vcur['acc'])]], dtype=torch.float32, device=device)
                logp_gt = dist.log_prob(a_gt_t).sum(dim=-1, keepdim=True).item()
                self._ep_obs_transitions.append((cur_s, float(vcur['acc']), float(logp_gt), nxt_s, False))
            else:
                self._ep_unobs_transitions.append((cur_s, float(a_rl), float(logp_rl), nxt_s, bool(d)))

        self.frame = next_frame
        return done_any, align_reward



    def position_update_PPO_control(self, next_frame, veh_id, vehicle_state, new_velocity):
        old_theta = vehicle_state['theta']
        r = self.circle_radius
        new_theta = old_theta + new_velocity * 0.04 / r
        if new_theta >= 2 * np.pi:
            new_theta = new_theta - 2 * np.pi
        self.vehicles_in_sim[next_frame][veh_id] = {
            'id': int(veh_id),
            'observed': vehicle_state["observed"],
            'idm_parameter': vehicle_state['idm_parameter'],
            'theta': float(new_theta),
            "v": float(new_velocity),
            'acc': float(vehicle_state['acc']),
            'relationships': {
                'preceding': None,
                'following': None
            }
        }
    def load_unobs_multi(self, start_frame, unobserved_ids, id_to_generated):
        """
        unobserved_ids: list[int]
        id_to_generated: dict[int, (theta, v)]
        """
        # 删除这些车在所有帧里的原始记录
        for frame in list(self.vehicles_in_sim.keys()):
            for uid in unobserved_ids:
                self.vehicles_in_sim[frame].pop(uid, None)

        # 在 start_frame 写入生成状态
        for uid in unobserved_ids:
            theta, v = id_to_generated[uid]
            self.vehicles_in_sim[start_frame][uid] = {
                'id': int(uid),
                'observed': False,
                'theta': float(theta),
                'v': float(v),
                'acc': 0.0,
                'idm_parameter': {'v0': None, 'a_max': None, 'b': None, 'T': None, 's0': None, 's_alpha': None},
                'relationships': {'preceding': None, 'following': None}
            }

        # 重建这一帧的前后关系
        self.update_surrounding(start_frame)

    def update_surrounding(self, frame):
        vehicles = self.vehicles_in_sim.get(frame, {})
        if not vehicles:
            return vehicles

        # 清空关系
        for v in vehicles.values():
            v['relationships'] = {'preceding': None, 'following': None}

        # 按角度升序排一圈
        ordered = sorted(vehicles.items(), key=lambda kv: kv[1]['theta'])
        n = len(ordered)
        r = self.circle_radius

        def long_distance(theta_from, theta_to):
            # 最短圆环角差 -> 与你现有口径一致的弧长（2 * Δθ * r）
            dtheta = abs((theta_to - theta_from + math.pi) % (2 * math.pi) - math.pi)
            return 2 * dtheta * r

        for i, (vid, v) in enumerate(ordered):
            pre_vid, pre_v = ordered[(i + 1) % n]
            fol_vid, fol_v = ordered[(i - 1) % n]
            v['relationships']['preceding'] = (int(pre_vid), long_distance(v['theta'], pre_v['theta']))
            v['relationships']['following'] = (int(fol_vid), long_distance(fol_v['theta'], v['theta']))

        return vehicles

    def determine_relationship(self, subject_vehicle, other_vehicle, category):
        r = self.circle_radius
        if other_vehicle['theta'] >= subject_vehicle['theta']:
            long_dis = 2 * (other_vehicle['theta'] - subject_vehicle['theta']) * r
        else:
            long_dis = 2 * (2 * np.pi - subject_vehicle['theta'] + other_vehicle['theta']) * r
        relationships = subject_vehicle['relationships']
        relationships[category] = (other_vehicle["id"], long_dis)

    def load_unobs(self, start_frame,unobserved_vehicle_id,unobserved_state_pred):
        """
        Updates self.vehicles_in_sim to remove the unobserved vehicle's data 
        and replace it with the generated data from the state generator.
        
        Parameters:
        - unobserved_vehicle_id: ID of the unobserved vehicle.
        - unobserved_state_pred: Predicted state tensor for the unobserved vehicle.
        """
        for frame in self.vehicles_in_sim:
            if unobserved_vehicle_id in self.vehicles_in_sim[frame]:
                # Remove the original unobserved vehicle data
                del self.vehicles_in_sim[frame][unobserved_vehicle_id]
                
        # Replace with generated data
        self.vehicles_in_sim[start_frame][unobserved_vehicle_id] = {
            'id': unobserved_vehicle_id,
            'observed': False,  # This vehicle is unobserved
            'theta': unobserved_state_pred[0, 0].item(),
            'v': unobserved_state_pred[0, 1].item(),
            'acc': 0.0,  # Acceleration is set to 0
            'idm_parameter': {
                'v0': None,
                'a_max': None,
                'b': None,
                'T': None,
                's0': None,
                's_alpha': None
            },
            'relationships': {
                'preceding': None,
                'following': None
            }
        }

    def initialize_environment(self, start_frame):
        self.reset()
        self.frame = start_frame
        self.vehicles_in_sim = {}

        end_frame = start_frame + FRAMES_TO_PLAY + 1

        frames_to_load = self.vehicle_data['Frame'].unique()
        frames_to_load = frames_to_load[(frames_to_load >= start_frame) & (frames_to_load < end_frame)]

        for frame in frames_to_load:
            self.vehicles_in_sim[frame] = {}
            for _, row in self.vehicle_data[self.vehicle_data['Frame'] == frame].iterrows():
                vehicle_id = int(row['Vehicle ID'])
                self.vehicles_in_sim[frame][vehicle_id] = {
                    'id': int(vehicle_id),
                    'observed': True,
                    'theta': float(row['Theta']),
                    'v': float(row['Velocity']),
                    'acc': float(row['acc']),
                    'idm_parameter': {
                        'v0': float(row['v0']),
                        'a_max': float(row['a_max']),
                        'b': float(row['b']),
                        'T': float(row['T']),
                        's0': float(row['s0']),
                        's_alpha': float(row['s_alpha'])
                    },
                    'relationships': {
                        'preceding': eval(row['Preceding']) if row['Preceding'] else None,
                        'following': eval(row['Following']) if row['Following'] else None
                    }
                }

        # build speed zones for every frame we might step into
        self.speed_zones_per_frame = {}
        df = self.speed_limits
        zones_in_window = df[df["Frame"].between(start_frame, end_frame - 1)]
        grouped = zones_in_window.groupby("Frame")
        for f in range(start_frame, end_frame):
            if f not in grouped.indices:
                raise ValueError(
                    f"Missing speed zones for frame {f}. "
                    f"Check {speed_limits_path} or preprocessing."
                )
            zdf = grouped.get_group(f)
            self.speed_zones_per_frame[f] = [
                (float(row["Start Angle"]), float(row["End Angle"]), float(row["Speed Limit"]))
                for _, row in zdf.iterrows()
            ]

        # choose observed vs unobserved ids for this episode
        ids = list(self.vehicles_in_sim[start_frame].keys())
        ids = ids[:self.number_of_vehicle] if len(ids) > self.number_of_vehicle else ids
        random.shuffle(ids)
        observed_set = set(ids[:self.n_observed])
        for vid in ids:
            self.vehicles_in_sim[start_frame][vid]['observed'] = (vid in observed_set)

        # --- reset episode-scoped buffers ---
        self._ep_obs_transitions = []
        self._ep_unobs_transitions = []
        self._episode_align_sum = 0.0
        self._episode_steps = 0
        self._num_unobserved = sum(1 for vid in ids if not self.vehicles_in_sim[start_frame][vid]['observed'])

    def load_data(self, start_frame):

        self.reset()
        self.frame = start_frame
        self.vehicles_in_sim = {}

        end_frame = start_frame + FRAMES_TO_PLAY + 1  # still needs the very next frame
        frames_to_load = self.vehicle_data['Frame'].unique()
        frames_to_load = frames_to_load[(frames_to_load >= start_frame) & (frames_to_load < end_frame)]

        for frame in frames_to_load:
            self.vehicles_in_sim[frame] = {}
            for index, row in self.vehicle_data[self.vehicle_data['Frame'] == frame].iterrows():
                vehicle_id = int(row['Vehicle ID'])
                self.vehicles_in_sim[frame][vehicle_id] = {
                    'id': int(vehicle_id),
                    'observed': True,
                    'theta': float(row['Theta']),
                    'v': float(row['Velocity']),
                    'acc': float(row['acc']),
                    'idm_parameter': {
                        'v0': float(row['v0']),
                        'a_max': float(row['a_max']),
                        'b': float(row['b']),
                        'T': float(row['T']),
                        's0': float(row['s0']),
                        's_alpha': float(row['s_alpha'])
                    },
                    'relationships': {
                        'preceding': eval(row['Preceding']) if row['Preceding'] else None,
                        'following': eval(row['Following']) if row['Following'] else None
                    }
                }

        self.speed_zones_per_frame = {}
        frames_to_load = self.speed_limits['Frame'].unique()
        frames_to_load = frames_to_load[(frames_to_load >= start_frame) & (frames_to_load < end_frame)]

        for frame in frames_to_load:
            self.speed_zones_per_frame[frame] = []
            for index, row in self.speed_limits[self.speed_limits['Frame'] == frame].iterrows():
                start_angle = float(row['Start Angle'])
                end_angle = float(row['End Angle'])
                speed_limit = float(row['Speed Limit'])
                self.speed_zones_per_frame[frame].append((start_angle, end_angle, speed_limit))
    # single_lane_env.py  (class CircularRaceTrackEnv 内)
    def flush_episode_memory(self):
        """
        在 episode 结束时将所有暂存 transition 推入回放池。
        现在的策略（按你的新需求）：
        - 所有车辆奖励相同，均为：
            avg_all_reward = (episode 对齐奖励和) / (步数 × 所有车辆数)
        - 观测车的 done 仍为 False；不观测车的 done 保持其各自标记
        """
        if self._episode_steps == 0:
            return

        # 所有车辆数量：采用配置的 self.number_of_vehicle（与 initialize_environment 的截断一致）
        num_all = max(1, int(getattr(self, 'number_of_vehicle', 1)))
        denom   = max(1, self._episode_steps * num_all)
        avg_all_reward = float(self._episode_align_sum) / float(denom)

        # 观测车：统一奖励 = avg_all_reward，done=False
        for s, a, lp, s2, _ in self._ep_obs_transitions:
            self.replay_memory.push(s, [a], lp, avg_all_reward, s2, False)

        # 不观测车：统一奖励 = avg_all_reward，done 按各自记录
        for s, a, lp, s2, d in self._ep_unobs_transitions:
            self.replay_memory.push(s, [a], lp, avg_all_reward, s2, d)

        # 清空
        self._ep_obs_transitions.clear()
        self._ep_unobs_transitions.clear()
        self._episode_align_sum = 0.0
        self._episode_steps = 0
        self._num_unobserved = 0

    def generator_obs(self, initial_frame, unobserved_vehicle_id):
        # this function is the get the unobserved vehicle info as generator input 
        # 统一成集合，便于后续 in 检查
        if isinstance(unobserved_vehicle_id, int):
            unobs_ids = {unobserved_vehicle_id}
        else:
            unobs_ids = set(unobserved_vehicle_id)

        generatorinput = []
        for vehicle_id, vehicle_state in self.vehicles_in_sim[initial_frame].items():
            if vehicle_id not in unobs_ids:
                generatorinput.extend([vehicle_state['theta'], vehicle_state['v']])
        # 填充或截断到 10 维（5 辆车 × 2）
        while len(generatorinput) < 10:
            generatorinput.append(0.0)
        generatorinput = generatorinput[:10]
        return generatorinput
    
    def class_to_size(self, vehicle_class):
        return 15 if vehicle_class == "Truck" else 4.5
    
    def plot_frame(self, frame,generate_vehicle_id,generated_vehicle_speed):
        if frame not in self.vehicles_in_sim:
            print(f"No data for frame {frame}")
            return

        data = self.vehicles_in_sim[frame]
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_ylim(0, self.circle_radius + 5)
        
        # Plot speed limit zones
        if frame in self.speed_zones_per_frame:
            colors = ['b', 'g', 'r', 'm', 'y']
            for idx, (start_angle, end_angle, speed_limit) in enumerate(self.speed_zones_per_frame[frame]):
                theta_zone = np.linspace(start_angle, end_angle, 100)
                r_zone_inner = np.zeros_like(theta_zone)
                r_zone_outer = np.full_like(theta_zone, self.circle_radius)
                ax.fill_between(theta_zone, r_zone_inner, r_zone_outer, color=colors[idx % len(colors)], alpha=0.3, label=f"Speed Limit: {speed_limit:.1f} m/s")
                # Add annotations for speed limits
                mid_angle = (start_angle + end_angle) / 2
                ax.text(mid_angle, self.circle_radius / 2, f"{speed_limit:.1f} m/s", color=colors[idx % len(colors)],
                        horizontalalignment='center', verticalalignment='center')
        
        # Plot circular road
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.full_like(theta, self.circle_radius)
        ax.plot(theta, r, 'k-')
        
        # Function to plot curved rectangles
        def plot_vehicle(ax, theta, length, label):
            half_length_angle = length / (2 * np.pi * self.circle_radius) * 2 * np.pi
            theta_left = theta - half_length_angle
            theta_right = theta + half_length_angle
            ax.plot([theta_left, theta_right], [self.circle_radius, self.circle_radius], 'o-', label=label)

        # Plot vehicles
        for vehicle_id, vehicle in data.items():
            theta_v = vehicle['theta']
            length = self.class_to_size( "Car")  # Example: Alternate between truck and car
            plot_vehicle(ax, theta_v, length, f"Vehicle {vehicle_id}")
        
        plt.legend()
        plt.title(f"generated vehicle id {generate_vehicle_id},generated vehicle speed{generated_vehicle_speed}")
        save_dir = "generated_frames"  # Define a valid directory
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/frame_{frame}.png")
        plt.close(fig)
