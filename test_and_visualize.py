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

def vis_2D():
    test = torch.rand(64, 2).to(device)
    X = test.view(1, 64, 2)
    discrepancy = L2discrepancy(X)
    print("discrepancy:", discrepancy)

    points = np.load('outputs/dim_2/best_points.npy', allow_pickle=True)[6]
    points = torch.from_numpy(points)
    points = points.view(1, 64, 2)

    batch = torch.arange(1).unsqueeze(-1).to(device)
    batch = batch.repeat(1, 64).flatten()
    edge_index = radius_graph(test, r=0.2, loop=True, batch=batch).to(device)
    model = MPMC_net(2, 128, 3, 64, 1, 0.2, 'L2', [1], 15).to(device)
    model.load_state_dict(torch.load("outputs/dim_2/best_model.pth"))
    model.eval()
    loss, points = model.forward(test, edge_index, batch)
    discrepancy = L2discrepancy(points)
    print("discrepancy after GNN:", discrepancy.item())

    test_points = test.numpy()
    final_points = points.detach().cpu().squeeze(0).numpy()

    # Plot the initial random points and the final points side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Initial points visualization
    axs[0].scatter(test_points[:, 0], test_points[:, 1], c='blue', alpha=0.7)
    axs[0].set_title("Initial Random Points")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_aspect('equal', 'box')

    # Final points visualization
    axs[1].scatter(final_points[:, 0], final_points[:, 1], c='red', alpha=0.7)
    axs[1].set_title("Points After MPNN Transformation")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_aspect('equal', 'box')

    plt.tight_layout()
    plt.show()

def vis_3D():
    points = np.load('outputs/dim_2/nsamples_64.npy', allow_pickle=True)[0]
    points = torch.from_numpy(points)

    xyz = torch.randn(64, 3, device=device)
    # Normalize each row to have unit length
    points = xyz / xyz.norm(dim=1, keepdim=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')



    # Extract x, y, z
    x_coord = points[:,0]
    y_coord = points[:,1]
    z_coord = points[:,2]

    # Plot the sampled points
    ax.scatter(x_coord, y_coord, z_coord, c='blue', s=20, alpha=0.7)

    # Optionally: draw a sphere surface for reference
    # Create a wireframe of a unit sphere:
    u = torch.linspace(0, 2 * torch.pi, 30)
    v = torch.linspace(0, torch.pi, 30)
    u, v = torch.meshgrid(u, v, indexing='ij')
    u = u.numpy()
    v = v.numpy()

    xs = np.cos(u)*np.sin(v)
    ys = np.sin(u)*np.sin(v)
    zs = np.cos(v)

    ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.3)

    # Set equal aspect ratio for all axes
    max_radius = 1.0
    for direction in (-1, 1):
        for point in np.diag(direction * max_radius * np.ones(3)):
            ax.plot([point[0]], [point[1]], [point[2]], 'w')

    ax.set_box_aspect([1,1,1])  # Make the aspect ratio equal
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sampled Points on the Sphere')

    plt.show()

if __name__ == "__main__":
    vis_2D()
    # vis_3D()
