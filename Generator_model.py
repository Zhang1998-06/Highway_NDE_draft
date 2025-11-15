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
from utils import *
SEED = 42
DROP_OUT_P = 0.5
TRAJECTORIES =20000
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

# Set device for PyTorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)



class GeneratorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GeneratorModel, self).__init__()
        # ... (other layers remain unchanged)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_activation = nn.ReLU()
        self.theta_out = nn.Linear(hidden_dim, 1)   # Modified: 1 continuous output for theta
        self.speed_out = nn.Linear(hidden_dim, 1)   # Modified: 1 continuous output for speed

    def forward(self, x):
        h = self.hidden_activation(self.hidden(x))
        theta_norm = torch.sigmoid(self.theta_out(h))        # Modified: theta normalized to [0, 1]
        theta = theta_norm * MAX_THETA                    # Modified: scale to [0, 2*MAX_THETA]
        speed_norm = torch.sigmoid(self.speed_out(h))     # Modified: speed normalized to [0, 1]
        speed = speed_norm * MAX_SPEED                    # Modified: scale to [0, MAX_SPEED]
        return theta, speed
    