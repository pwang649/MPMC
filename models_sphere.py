import torch
import math
from torch import nn
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing, InstanceNorm

from test_and_visualize import fibonacci_sphere_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Helper to sample uniformly on a sphere (2-sphere in 3D).
# This uses the fact that sampling a 3D Gaussian and normalizing gives uniform direction.
def sample_sphere(n):
    # Draw samples from a normal distribution
    xyz = torch.randn(n, 3, device=device)
    # Normalize each row to have unit length
    xyz = xyz / xyz.norm(dim=1, keepdim=True)
    return xyz



class MPNN_layer(MessagePassing):
    def __init__(self, ninp, nhid):
        super(MPNN_layer, self).__init__()
        self.ninp = ninp
        self.nhid = nhid

        self.message_net_1 = nn.Sequential(nn.Linear(2 * ninp, nhid),
                                           nn.ReLU())
        self.message_net_2 = nn.Sequential(nn.Linear(nhid, nhid),
                                           nn.ReLU())
        self.update_net_1 = nn.Sequential(nn.Linear(ninp + nhid, nhid),
                                          nn.ReLU())
        self.update_net_2 = nn.Sequential(nn.Linear(nhid, nhid),
                                          nn.ReLU())
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
    def __init__(self, nhid, nlayers, nsamples, nbatch, radius, n_reference=1000):
        super(MPMC_net, self).__init__()
        # We now fix dim=3 for sampling on a sphere S^2.
        self.dim = 3
        self.enc = nn.Linear(self.dim, nhid)
        self.convs = nn.ModuleList([MPNN_layer(nhid, nhid) for _ in range(nlayers)])
        self.dec = nn.Linear(nhid, self.dim)
        self.nlayers = nlayers
        self.nbatch = nbatch
        self.nsamples = nsamples

        # Initial points on the sphere:
        self.x = sample_sphere(nsamples * nbatch)
        batch = torch.arange(nbatch, device=device).unsqueeze(-1)
        batch = batch.repeat(1, nsamples).flatten()
        self.batch = batch
        self.edge_index = radius_graph(self.x, r=radius, loop=True, batch=batch).to(device)

        # Precompute a reference set Y of uniformly distributed points on the sphere.
        self.Y = fibonacci_sphere_samples(64)

        # Precompute kernel values for Y to avoid doing this repeatedly:
        with torch.no_grad():
            self.K_YY = self.kernel(self.Y, self.Y).mean()

    def kernel(self, A, B):
        # Dot-product-based kernel
        sigma = 0.5
        dot_prod = (A[:, None, :] * B[None, :, :]).sum(dim=-1)
        return torch.exp((dot_prod - 1) / (2 * sigma ** 2))

    def spherical_discrepancy(self, X):
        # X: [batch, N, 3]
        B, N, d = X.shape

        # Compute MMD per batch and accumulate
        MMD_sum = 0.0
        for b in range(B):
            Xb = X[b]  # shape [N,3]
            K_XX_b = self.kernel(Xb, Xb).mean()
            K_XY_b = self.kernel(Xb, self.Y).mean()
            # MMD² = E[K(X,X)] - 2 E[K(X,Y)] + E[K(Y,Y)]
            MMD_b = K_XX_b - 2 * K_XY_b + self.K_YY
            MMD_sum += MMD_b

        return MMD_sum / B

    def forward(self):
        X = self.x
        edge_index = self.edge_index

        # Encode
        X = self.enc(X)
        # MPNN layers
        for i in range(self.nlayers):
            X = self.convs[i](X, edge_index, self.batch)
        # Decode
        X = self.dec(X)
        # Project onto sphere
        X = X / X.norm(dim=-1, keepdim=True)

        X = X.view(self.nbatch, self.nsamples, self.dim)
        loss = self.spherical_discrepancy(X)
        return loss, X
