from matplotlib import pyplot as plt
from torch_cluster import radius_graph

from models import *
import torch
import torch.optim as optim
import argparse

from models_GAT import GAT_net
from models_GCN import GCN_net
from utils import L2discrepancy, hickernell_all_emphasized

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # Data preparation
    data = torch.rand(args.nsamples * args.nbatch, args.dim).to(device)
    batch = torch.arange(args.nbatch).unsqueeze(-1).to(device)
    batch = batch.repeat(1, args.nsamples).flatten()
    edge_index = radius_graph(data, r=args.radius, loop=True, batch=batch).to(device)

    data_test = torch.rand(args.nsamples * args.nbatch, args.dim).to(device)
    batch_test = torch.arange(args.nbatch).unsqueeze(-1).to(device)
    batch_test = batch_test.repeat(1, args.nsamples).flatten()
    edge_index_test = radius_graph(data_test, r=args.radius, loop=True, batch=batch_test).to(device)

    # Initialize models
    models = {
        "MPNN": MPMC_net(args.dim, args.nhid, args.nlayers, args.nsamples, args.nbatch,
                         args.radius, args.loss_fn, args.dim_emphasize, args.n_projections).to(device),
        "GCN": GCN_net(args.dim, args.nhid, args.nlayers, args.nsamples, args.nbatch,
                       args.radius, args.loss_fn, args.dim_emphasize, args.n_projections).to(device),
        "GAT": GAT_net(args.dim, args.nhid, args.nlayers, args.nsamples, args.nbatch,
                       args.radius, args.loss_fn, args.dim_emphasize, args.n_projections).to(device),
    }

    optimizers = {
        model_name: optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for model_name, model in models.items()
    }

    # Tracking results
    mean_discrepancy_logs = {model_name: [] for model_name in models}
    std_discrepancy_logs = {model_name: [] for model_name in models}

    for epoch in range(args.epochs):
        for model_name, model in models.items():
            model.train()
            optimizers[model_name].zero_grad()
            loss, X = model(data, edge_index, batch)
            loss.backward()
            optimizers[model_name].step()

            # Evaluate every 10 epochs
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    loss_test, X_test = model(data_test, edge_index_test, batch_test)
                    if args.loss_fn == 'L2':
                        batched_discrepancies = L2discrepancy(X_test.detach())
                    elif args.loss_fn == 'approx_hickernell':
                        batched_discrepancies = hickernell_all_emphasized(X_test.detach(), args.dim_emphasize)
                    else:
                        raise ValueError("Loss function not implemented")

                    mean_discrepancy = torch.mean(batched_discrepancies).item()
                    std_discrepancy = torch.std(batched_discrepancies).item()

                    mean_discrepancy_logs[model_name].append(mean_discrepancy)
                    std_discrepancy_logs[model_name].append(std_discrepancy)

                    print(f"Epoch {epoch}, Model {model_name}, Mean Discrepancy: {mean_discrepancy}, Std: {std_discrepancy}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for model_name in models:
        epochs_range = range(0, args.epochs, 10)
        mean_logs = mean_discrepancy_logs[model_name]
        std_logs = std_discrepancy_logs[model_name]
        plt.plot(epochs_range, mean_logs, label=f"{model_name} Mean")
        plt.fill_between(epochs_range,
                         [m - s for m, s in zip(mean_logs, std_logs)],
                         [m + s for m, s in zip(mean_logs, std_logs)],
                         alpha=0.2, label=f"{model_name} Std")

    plt.xlabel("Epochs")
    plt.ylabel("Discrepancy (Mean ± Std)")
    plt.title("Mean and Std of Discrepancy vs. Epochs for MPNN, GCN, and GAT")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot results
    plt.figure(figsize=(10, 6))
    for model_name, logs in mean_discrepancy_logs.items():
        plt.plot(range(0, args.epochs, 10), logs, label=model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Discrepancy")
    plt.title("Mean Discrepancy vs. Epochs for MPNN, GCN, and GAT")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='number of samples')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of GNN nlayers')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight_decay')
    parser.add_argument('--nhid', type=int, default=128,
                        help='number of hidden features of GNN')
    parser.add_argument('--nbatch', type=int, default=128,
                        help='number of point sets in batch')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs')
    parser.add_argument('--start_reduce', type=int, default=1000,
                        help='when to start lr decay')
    parser.add_argument('--radius', type=float, default=0.2,
                        help='radius for nearest neighbor GNN graph')
    parser.add_argument('--nsamples', type=int, default=64,
                        help='number of samples')
    parser.add_argument('--dim', type=int, default=2,
                        help='dimension of points')
    parser.add_argument('--loss_fn', type=str, default='L2',
                        help='which loss function to use. Choices: ["L2","approx_hickernell"]')
    parser.add_argument('--dim_emphasize', type=list, default=[1],
                        help='if loss_fn set to "approx_hickernell", specify which dimensionality to emphasize.'
                             'Note: It is not the coordinate of the points, but the dimension of the'
                             'projections, i.e., seeting dim_emphasize = [1,3] puts an emphasize'
                             'on 1-dimensional and 3-dimensional projections. Cannot emphasize all'
                             'dimensionalities.')
    parser.add_argument('--n_projections', type=int, default=15,
                        help='number of projections for approx_hickernell')

    args = parser.parse_args()
    train(args)

