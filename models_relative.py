import torch
import math
from torch import nn
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing, InstanceNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MPNN_layer(MessagePassing):
    def __init__(self, ninp, nhid):
        super(MPNN_layer, self).__init__()
        self.ninp = ninp
        self.nhid = nhid

        self.message_net_1 = nn.Sequential(
            nn.Linear(2 * ninp, nhid),
            nn.ReLU()
        )
        self.message_net_2 = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU()
        )
        self.update_net_1 = nn.Sequential(
            nn.Linear(ninp + nhid, nhid),
            nn.ReLU()
        )
        self.update_net_2 = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU()
        )
        self.norm = InstanceNorm(nhid)

    def forward(self, x, edge_index, batch):
        x = self.propagate(edge_index, x=x)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j):
        message = self.message_net_1(torch.cat((x_i, x_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return update


class MPMC_net(nn.Module):
    def __init__(self, dim, nhid, nlayers, radius, loss_fn, dim_emphasize, n_projections):
        super(MPMC_net, self).__init__()
        self.enc = nn.Linear(dim, nhid)
        self.convs = nn.ModuleList()
        for i in range(nlayers):
            self.convs.append(MPNN_layer(nhid, nhid))
        self.dec = nn.Linear(nhid, dim)
        self.nlayers = nlayers
        self.dim = dim
        self.n_projections = n_projections
        self.dim_emphasize = torch.tensor(dim_emphasize).long() if dim_emphasize is not None else None
        self.radius = radius

        # Choose loss function
        if loss_fn == 'L2':
            self.loss_fn = self.L2discrepancy
        elif loss_fn == 'approx_hickernell':
            if dim_emphasize is not None:
                assert torch.max(self.dim_emphasize) <= dim
                self.loss_fn = self.approx_hickernell
            else:
                raise ValueError("dim_emphasize cannot be None for approx_hickernell.")
        else:
            raise ValueError("Loss function not implemented")

    def approx_hickernell(self, X):
        # X is [nbatch, nsamples, dim]
        nbatch, nsamples, dim = X.size()
        disc_projections = torch.zeros(nbatch).to(device)

        for _ in range(self.n_projections):
            # Sample among non-emphasized dimensions
            mask = torch.ones(dim, dtype=torch.bool)
            mask[self.dim_emphasize - 1] = False
            remaining_dims = torch.arange(1, dim + 1)[mask]
            projection_dim = remaining_dims[torch.randint(low=0, high=remaining_dims.size(0), size=(1,))].item()
            projection_indices = torch.randperm(dim)[:projection_dim]
            disc_projections += self.L2discrepancy(X[:, :, projection_indices])

            # Sample among emphasized dimensions
            emphasized_dims = torch.arange(1, dim + 1)[self.dim_emphasize - 1]
            projection_dim = emphasized_dims[torch.randint(low=0, high=emphasized_dims.size(0), size=(1,))].item()
            projection_indices = torch.randperm(dim)[:projection_dim]
            disc_projections += self.L2discrepancy(X[:, :, projection_indices])

        return disc_projections

    def L2discrepancy(self, x):
        # x: [nbatch, nsamples, dim]
        nbatch, N, dim = x.size()
        prod1 = 1. - x**2
        prod1 = torch.prod(prod1, dim=2)  # [nbatch, nsamples]
        sum1 = torch.sum(prod1, dim=1)    # [nbatch]

        pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :]) # [nbatch, nsamples, nsamples, dim]
        product = torch.prod(1 - pairwise_max, dim=3)  # [nbatch, nsamples, nsamples]
        sum2 = torch.sum(product, dim=(1, 2))           # [nbatch]

        one_dive_N = 1. / N
        out = torch.sqrt((3.**(-dim)) - one_dive_N * (2.**(1.-dim)) * sum1 + (1. / (N**2)) * sum2)
        return out

    def forward(self, X, batch):
        # X: [N, dim], batch: [N]
        # Here N can vary depending on how many samples are given
        nbatch = (batch.max().item() + 1)
        N = X.size(0)
        if N % nbatch != 0:
            raise ValueError("Number of samples is not divisible by nbatch.")
        nsamples = N // nbatch

        # Build graph dynamically based on current input
        edge_index = radius_graph(X, r=self.radius, loop=True, batch=batch)

        # Encode
        X = self.enc(X)

        # Message passing layers
        for conv in self.convs:
            X = conv(X, edge_index, batch)

        # Decode and ensure samples are in [0,1] range
        X = torch.sigmoid(self.dec(X))  # [N, dim]

        # Reshape to [nbatch, nsamples, dim] for discrepancy calculation
        X_reshaped = X.view(nbatch, nsamples, self.dim)
        loss = torch.mean(self.loss_fn(X_reshaped))
        return loss, X_reshaped
