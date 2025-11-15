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

# Set random seeds for reproducibility
# the thing is th
SEED = 42
DROP_OUT_P = 0.5
TRAJECTORIES = 800
FRAMES_TO_PLAY = 500
SEGMENT_LEN = 100
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 128

# Hyperparameters for PPO
LEARNING_RATE_ACTOR = 5e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.98
LAMBDA = 0.95
EPOCHS = 2
EPS_CLIP = 0.1
HIDDEN_DIM = 128

# Environment and data paths
segmentPath = f"toy_study/data/_{SEGMENT_LEN}_{SEED}_{DROP_OUT_P}"
vehicle_data_path = f"data/highd/sim/_{SEED}_{DROP_OUT_P}.csv"
speed_limits_path = vehicle_data_path.replace(".csv", "_speed_limits.csv")
output_path = "data/highd/sim/combined_trajectories.csv"

# Set device for PyTorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

def lmap(v: float, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

def load_vehicle_data():
    return pd.read_csv(vehicle_data_path)

def load_speed_limits():
    return pd.read_csv(speed_limits_path)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_array = np.array(advantage_list, dtype=np.float32)
    return torch.tensor(advantage_array)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = lmap(torch.tanh(self.fc_mu(x)), [-1, 1], [-1.1, 0.5])
        std = torch.clamp(F.softplus(self.fc_std(x)), min=1e-8, max=0.4)
        return mu, std

class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, batch_size):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.batch_size = batch_size

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        mu = torch.clamp(mu, min=-1.1, max=0.5)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        action = torch.clamp(action, min=-1.1, max=0.5)

        log_prob = action_dist.log_prob(action)

        # Convert tensors to numpy arrays, detaching from the graph first
        action = action.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()

        return action, log_prob

    def update_controller(self, replay_memory):
        if len(replay_memory) < self.batch_size:
            return

        # Sample a batch of transitions
        state_batch, action_batch, next_state_batch, reward_batch, done_mask = \
            replay_memory.sample(self.batch_size)

        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        done_mask = torch.from_numpy(done_mask).float().to(self.device)

        # Compute the target and advantage
        td_target = reward_batch.view(-1, 1) + self.gamma * self.critic(next_state_batch) * (1 - done_mask.view(-1, 1))
        td_delta = td_target - self.critic(state_batch)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        for _ in range(self.epochs):
            mu, std = self.actor(state_batch)
            mu = torch.clamp(mu, min=-1.1, max=0.5)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(action_batch)
            ratio = torch.exp(log_probs - log_probs.detach())  # Reuse the log_probs here as a placeholder
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(state_batch), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            '''
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5) 
            '''           
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Ensure the reward is stored as a scalar or a consistent shape
        if isinstance(reward, np.ndarray) or isinstance(reward, list):
            reward = np.array(reward).item()  # Convert to scalar if possible
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_mask = zip(*random.sample(self.memory, batch_size))
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        # Convert rewards to a consistent shape
        reward_batch = np.array([np.array(r).item() if isinstance(r, np.ndarray) or isinstance(r, list) else r for r in reward_batch])
        next_state_batch = np.array(next_state_batch)
        done_mask = np.array(done_mask)
        return state_batch, action_batch, next_state_batch, reward_batch, done_mask

    def __len__(self):
        return len(self.memory)

class CircularRaceTrackEnv:
    def __init__(self, model, circular_radius=10, number_of_vehicle=10):
        self.action_space = 1  # continuous action space
        self.n_vehicles = number_of_vehicle
        self.observation_space = 4  # [theta, velocity,speed_limit, preceding_vehicle_distance]
        self.vehicles_in_sim = {}
        self.simulation_duration = 10000
        self.model = model
        self.Speed_reward_weight = 1
        self.circle_radius = circular_radius
        self.frame = 0
        self.vehicle_data = load_vehicle_data()
        self.speed_limits = load_speed_limits()
        self.agent = PPOContinuous(state_dim=self.observation_space, hidden_dim=HIDDEN_DIM, action_dim=self.action_space,
                         actor_lr=LEARNING_RATE_ACTOR, critic_lr=LEARNING_RATE_CRITIC, lmbda=LAMBDA, epochs=EPOCHS,
                         eps=EPS_CLIP, gamma=GAMMA, device=device, batch_size=BATCH_SIZE)
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    def reset(self):
        self.frame = 0
        self.vehicles_in_sim = {}
        
    def get_speed_limit(self, theta, frame):
        for start_angle, end_angle, speed_limit in self.speed_zones_per_frame[frame]:
            if start_angle < end_angle:
                if start_angle <= theta < end_angle:
                    return speed_limit
            else:  # Wrap around case
                if theta >= start_angle or theta < end_angle:
                    return speed_limit

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

        # Get the preceding vehicle distance
        preceding_distance, _ = check_ROI()

        # Get the velocity of the ego vehicle
        ego_velocity = self.vehicles_in_sim[frame][veh_id]['v']

        # Get the velocity of the preceding vehicle
        preceding_vehicle_id = self.vehicles_in_sim[frame][veh_id]['relationships'].get('preceding')[0] if self.vehicles_in_sim[frame][veh_id]['relationships'].get('preceding') else None
        preceding_velocity = self.vehicles_in_sim[frame][preceding_vehicle_id]['v'] if preceding_vehicle_id is not None else 0

        # Calculate relative speed with the preceding vehicle
        relative_speed_with_preceding = ego_velocity - preceding_velocity

        # State representation
        state = [
            ego_velocity,
            speed_limit,
            preceding_distance,
            relative_speed_with_preceding
        ]

        return state

    def step(self, ppo_vehicle_id):
        frame = self.frame
        next_frame = frame + 1

        if next_frame not in self.vehicles_in_sim:
            self.vehicles_in_sim[next_frame] = {}

        # Iterate over vehicles in the current frame
        for veh_id, vehicle_state in self.vehicles_in_sim[frame].items():
            speed_limit = self.get_speed_limit(vehicle_state['theta'], frame)
            state = self.obs(frame, veh_id, speed_limit)

            if veh_id == ppo_vehicle_id:
                # PPO Agent Control with continuous actions
                action, log_prob = self.agent.take_action(state)
                vehicle_state['acc'] = action[0]
                vehicle_state["observed"] = False
                new_velocity = self.execute_action(vehicle_state['v'], action[0], speed_limit)
                self.position_update_PPO_control(next_frame, veh_id, vehicle_state, new_velocity)


        vehicles_this_frame = self.update_surrounding(next_frame)
        self.vehicles_in_sim[next_frame].update(vehicles_this_frame)

        Crash=False
        done=False
        # Compute rewards and store transitions
        reward_step=0
        for veh_id, vehicle_state in self.vehicles_in_sim[next_frame].items():
            next_speed_limit = self.get_speed_limit(vehicle_state['theta'], next_frame)
            current_speed_limit = self.get_speed_limit(self.vehicles_in_sim[frame][veh_id]['theta'], frame)
            next_state = self.obs(next_frame, veh_id,next_speed_limit)
            current_state = self.obs(frame, veh_id,current_speed_limit)
            current_action, log_prob = self.agent.take_action(current_state)
            #print(current_action)
            Crash=self.crashes(vehicle_state, next_frame, veh_id)
            if Crash==True:
                done=True
            if veh_id == ppo_vehicle_id:
                '''
                # Reward for PPO-controlled vehicle based on speed limit from dataset
                RL_reward = 1-abs(next_state[0] -next_speed_limit )/3 if abs(next_state[0] -next_speed_limit ) < 3 else 0
                RL_reward = RL_reward if Crash==False else -10
                # Save transition for PPO agent
                self.replay_memory.push(current_state, current_action[0], RL_reward, next_state, Crash)
                reward_step+=RL_reward 
                #print("sign1")
                #print(RL_reward)'''
            else:
                # Observed Vehicle Reward based on action difference

                dataset_acceleration = self.vehicles_in_sim[frame][veh_id]['acc']
                #print("dataset_acc:", dataset_acceleration)   
                #print("current_acc:", current_action[0]) 
                supervised_reward = np.clip(1 - abs(current_action[0] - dataset_acceleration) / 1.6, 0, 1)  # Use only the first element if needed
                #print("reward")
                #print("supervised_reward",supervised_reward)
                # Save transition for observed vehicle
                self.replay_memory.push(current_state, current_action[0], supervised_reward, next_state, False)
                reward_step+=supervised_reward
                #print("sign2")
                #print(supervised_reward)

        self.frame = next_frame
        return done,reward_step



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

    def update_surrounding(self, frame):
        vehicles = self.vehicles_in_sim.get(frame, {})
        for veh_id, veh_state in vehicles.items():
            veh_state['relationships'] = {
                'preceding': None,
                'following': None
            }
            n_vehicles = len(vehicles)
            preceding_id = (veh_id + 1) % n_vehicles
            following_id = (veh_id - 1) % n_vehicles
            self.determine_relationship(veh_state, vehicles[preceding_id], 'preceding')
        return vehicles

    def determine_relationship(self, subject_vehicle, other_vehicle, category):
        r = self.circle_radius
        if other_vehicle['theta'] >= subject_vehicle['theta']:
            long_dis = 2 * (other_vehicle['theta'] - subject_vehicle['theta']) * r
        else:
            long_dis = 2 * (2 * np.pi - subject_vehicle['theta'] + other_vehicle['theta']) * r
        relationships = subject_vehicle['relationships']
        relationships[category] = (other_vehicle["id"], long_dis)

    def initialize_environment(self, start_frame):
        self.reset()
        self.frame = start_frame
        self.vehicles_in_sim = {}

        end_frame = start_frame + FRAMES_TO_PLAY + 1

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

def collect_data_with_ppo(model, circular_radius, number_of_vehicle): # key 
    env = CircularRaceTrackEnv(model=model, circular_radius=circular_radius, number_of_vehicle=number_of_vehicle)
    all_trajectories = []
    return_list = []


    for trajectory_index in range(int(TRAJECTORIES)):
        min_frame = env.vehicle_data['Frame'].min()
        max_frame = env.vehicle_data['Frame'].max() - FRAMES_TO_PLAY - 1
        start_frame = random.choice(range(min_frame, max_frame))
        env.initialize_environment(start_frame)
        trajectory = {}
        ppo_vehicle_id = random.choice(list(env.vehicles_in_sim[start_frame].keys()))
        crash = False
        episode_return = 0
        
        for _ in range(FRAMES_TO_PLAY):
            if crash:
                break
            crash, reward = env.step(ppo_vehicle_id)
            trajectory[env.frame] = copy.deepcopy(env.vehicles_in_sim[env.frame])
            episode_return += reward
            env.agent.update_controller(env.replay_memory)
            
        return_list.append(float(episode_return))
        all_trajectories.append(trajectory)
        print(episode_return)
        


    print("Final return list:", return_list)
    
    pd.DataFrame(return_list, columns=['return']).to_csv('return_list_agent.csv', index=False)
    save_trajectories(all_trajectories)

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

if __name__ == "__main__":
    collect_data_with_ppo(model="generate_data", circular_radius=100, number_of_vehicle=5)
    ac_returns = pd.read_csv('return_list_agent.csv').squeeze().tolist()
    plot_mean_rewards(ac_returns, mean_number=10, smoothing_window=10)
