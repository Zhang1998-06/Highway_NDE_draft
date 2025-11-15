# policy.py
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Global seeds & PPO hparams (与训练脚本保持一致) =====
SEED = 42
DROP_OUT_P = 0.5
TRAJECTORIES = 800
FRAMES_TO_PLAY = 100
SEGMENT_LEN = 100
REPLAY_MEMORY_SIZE = 1000
BATCH_SIZE = 128

LEARNING_RATE_ACTOR = 5e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.98
LAMBDA = 0.95
EPOCHS = 2
EPS_CLIP = 0.1
HIDDEN_DIM = 128

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- torch 版本的线性映射（支持 tensor）----
def tmap(v: torch.Tensor, x_range, y_range) -> torch.Tensor:
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    return y0 + (v - x0) * (y1 - y0) / (x1 - x0)

def compute_advantage(gamma, lmbda, td_delta: torch.Tensor, device=None):
    # td_delta: [B,1]
    td_np = td_delta.detach().cpu().view(-1).numpy()
    advantage_list = []
    adv = 0.0
    for delta in td_np[::-1]:
        adv = gamma * lmbda * adv + float(delta)
        advantage_list.append(adv)
    advantage_list.reverse()
    adv_arr = np.array(advantage_list, dtype=np.float32)
    return torch.tensor(adv_arr, device=device if device is not None else None).unsqueeze(-1)

# ===== Networks =====
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # [B,1]

class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu_raw = torch.tanh(self.fc_mu(x))
        mu = tmap(mu_raw, [-1.0, 1.0], [-1.1, 0.5])  # 加速度范围
        std = torch.clamp(F.softplus(self.fc_std(x)), min=1e-4, max=0.4)
        return mu, std  # [B, A], [B, A]

# ===== PPO (修复 ratio 使用 old_log_prob；log_prob 按维度求和) =====
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
        self.action_dim = action_dim

    @torch.no_grad()
    def take_action(self, state):
        # state: list/np -> [1, state_dim]
        st = torch.tensor([state], dtype=torch.float32, device=self.device)
        mu, sigma = self.actor(st)
        mu = torch.clamp(mu, min=-1.1, max=0.5)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()                      # [1, A]
        action = torch.clamp(action, -1.1, 0.5)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # [1,1]
        return action.squeeze(0).detach().cpu().numpy(), log_prob.item()

    def update_controller(self, replay_memory):
        if len(replay_memory) < self.batch_size:
            return

        (state_batch, action_batch, next_state_batch,
         reward_batch, done_mask, old_log_prob_batch) = replay_memory.sample(self.batch_size)

        # to tensors
        state_batch = torch.from_numpy(state_batch).float().to(self.device)               # [B, S]
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)     # [B, S]
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device).unsqueeze(-1)  # [B,1]
        done_mask = torch.from_numpy(done_mask).float().to(self.device).unsqueeze(-1)        # [B,1]

        action_batch = torch.from_numpy(action_batch).float().to(self.device)
        if action_batch.ndim == 1:
            action_batch = action_batch.unsqueeze(-1)  # [B, A]
        old_log_prob_batch = torch.from_numpy(old_log_prob_batch).float().to(self.device).unsqueeze(-1)  # [B,1]

        # TD & advantage
        with torch.no_grad():
            v_next = self.critic(next_state_batch)
            v_cur = self.critic(state_batch)
            td_target = reward_batch + self.gamma * v_next * (1 - done_mask)
            td_delta = td_target - v_cur
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta, device=self.device)

        for _ in range(self.epochs):
            mu, std = self.actor(state_batch)
            mu = torch.clamp(mu, -1.1, 0.5)
            dist = torch.distributions.Normal(mu, std)
            new_log_prob = dist.log_prob(action_batch).sum(dim=-1, keepdim=True)  # [B,1]

            ratio = torch.exp(new_log_prob - old_log_prob_batch)  # [B,1]
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(self.critic(state_batch), td_target)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            # 可选：梯度裁剪
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

# ===== Replay Buffer（存 old_log_prob）=====
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push(self, state, action, old_log_prob, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # 保持 action 为 float 或 1D
        if isinstance(action, (list, np.ndarray)):
            action = float(np.array(action).reshape(-1)[0])
        # 保持 reward 为 float
        if isinstance(reward, (list, np.ndarray)):
            reward = float(np.array(reward).reshape(-1)[0])

        self.memory[self.pos] = (state, action, reward, next_state, done, float(old_log_prob))
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        s, a, r, s2, d, lp = zip(*batch)
        s = np.array(s, dtype=np.float32)
        a = np.array(a, dtype=np.float32)
        r = np.array(r, dtype=np.float32)
        s2 = np.array(s2, dtype=np.float32)
        d = np.array(d, dtype=np.float32)
        lp = np.array(lp, dtype=np.float32)
        return s, a, s2, r, d, lp

    def __len__(self):
        return len(self.memory)
