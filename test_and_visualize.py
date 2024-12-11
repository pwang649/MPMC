import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import L2discrepancy

# from models_sphere import MPMC_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


X = torch.rand(64,2)
discrepancy = L2discrepancy(X)
print("discrepancy:", discrepancy)

loaded_dict = np.load('outputs/dim_2/nsamples_64.npy', allow_pickle=True)
print(loaded_dict)
points = loaded_dict[6]
# model = MPMC_net(128, 3, 64, 3,
#                  0.35).to(device)
# model.load_state_dict(torch.load("model_state.pth"))
# model.eval()
# model.forward()
discrepancy = L2discrepancy(torch.from_numpy(points))
print("Spherical discrepancy (MMD):", discrepancy.item())
