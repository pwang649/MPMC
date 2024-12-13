import torch
from torch import nn
from torch_geometric.nn import GCNConv, InstanceNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN_net(nn.Module):
    def __init__(self, dim, nhid, nlayers, nsamples, nbatch, radius, loss_fn, dim_emphasize, n_projections):
        super(GCN_net, self).__init__()
        self.enc = nn.Linear(dim, nhid)
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GCNConv(nhid, nhid))
        self.dec = nn.Linear(nhid, dim)
        self.norm = InstanceNorm(nhid)
        self.nlayers = nlayers
        self.mse = torch.nn.MSELoss()
        self.nbatch = nbatch
        self.nsamples = nsamples
        self.dim = dim
        self.n_projections = n_projections
        self.dim_emphasize = torch.tensor(dim_emphasize).long()

        if loss_fn == 'L2':
            self.loss_fn = self.L2discrepancy
        elif loss_fn == 'approx_hickernell':
            if dim_emphasize is not None:
                assert torch.max(self.dim_emphasize) <= dim
                self.loss_fn = self.approx_hickernell
        else:
            raise ValueError("Loss function not implemented")

    def approx_hickernell(self, X):
        X = X.view(self.nbatch, self.nsamples, self.dim)
        disc_projections = torch.zeros(self.nbatch).to(device)

        for _ in range(self.n_projections):
            # Sample among non-emphasized dimensionality
            mask = torch.ones(self.dim, dtype=bool)
            mask[self.dim_emphasize - 1] = False
            remaining_dims = torch.arange(1, self.dim + 1)[mask]
            projection_dim = remaining_dims[torch.randint(low=0, high=remaining_dims.size(0), size=(1,))].item()
            projection_indices = torch.randperm(self.dim)[:projection_dim]
            disc_projections += self.L2discrepancy(X[:, :, projection_indices])
            # Sample among emphasized dimensionality
            remaining_dims = torch.arange(1, self.dim + 1)[self.dim_emphasize - 1]
            projection_dim = remaining_dims[torch.randint(low=0, high=remaining_dims.size(0), size=(1,))].item()
            projection_indices = torch.randperm(self.dim)[:projection_dim]
            disc_projections += self.L2discrepancy(X[:, :, projection_indices])

        return disc_projections

    def L2discrepancy(self, x):
        N = x.size(1)
        dim = x.size(2)
        prod1 = 1. - x ** 2.
        prod1 = torch.prod(prod1, dim=2)
        sum1 = torch.sum(prod1, dim=1)
        pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :])
        product = torch.prod(1 - pairwise_max, dim=3)
        sum2 = torch.sum(product, dim=(1, 2))
        one_dive_N = 1. / N
        out = torch.sqrt(3. ** -dim - one_dive_N * 2. ** (1. - dim) * sum1 + 1. / N ** 2. * sum2)
        return out

    def forward(self, X, edge_index, batch):
        X = self.enc(X)
        for conv in self.convs:
            X = conv(X, edge_index)
            X = self.norm(X, batch)
            X = torch.relu(X)
        X = torch.sigmoid(self.dec(X))  # Clamp values to [0, 1]
        X = X.view(self.nbatch, self.nsamples, self.dim)
        loss = torch.mean(self.loss_fn(X))
        return loss, X
