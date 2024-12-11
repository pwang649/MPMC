import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_cluster import radius_graph

from models import MPMC_net
from utils import L2discrepancy

# from models_sphere import MPMC_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


test = torch.rand(64, 2).to(device)
X = test.view(1, 64, 2)
discrepancy = L2discrepancy(X)
print("discrepancy:", discrepancy)

points = np.load('outputs/dim_2/nsamples_64.npy', allow_pickle=True)[0]
points = torch.from_numpy(points)
points = points.view(1, 64, 2)

batch = torch.arange(1).unsqueeze(-1).to(device)
batch = batch.repeat(1, 64).flatten()
edge_index = radius_graph(test, r=0.2, loop=True, batch=batch).to(device)
model = MPMC_net(2, 128, 3, 64, 1, 0.2, 'L2', [1], 15).to(device)
model.load_state_dict(torch.load("model_state.pth"))
model.eval()
loss, points = model.forward(test, edge_index, batch)
discrepancy = L2discrepancy(points)
print("discrepancy after GNN:", discrepancy.item())
