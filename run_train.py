import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
from utils import L2discrepancy, hickernell_all_emphasized
from models import *
from torch_cluster import radius_graph
from MLAP import MLAP_GIN, MLAP_Sum, MLAP_Weighted

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    data = torch.rand(args.nsamples * args.nbatch, args.dim).to(device)
    batch = torch.arange(args.nbatch).unsqueeze(-1).to(device)
    batch = batch.repeat(1, args.nsamples).flatten()
    edge_index = radius_graph(data, r=args.radius, loop=True, batch=batch).to(device)
    model = MPMC_net(args.dim, args.nhid, args.nlayers, args.nsamples, args.nbatch,
                     args.radius, args.loss_fn, args.dim_emphasize, args.n_projections).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = 10000.
    patience = 0

    ## could be tuned for better performance
    start_reduce = 100000
    reduce_point = 10

    Path('results/dim_' + str(args.dim)).mkdir(parents=True, exist_ok=True)
    Path('outputs/dim_' + str(args.dim)).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss, X = model(data, edge_index, batch)
        loss.backward()
        optimizer.step()

        if epoch % 100 ==0:
            y = X.clone()
            if args.loss_fn == 'L2':
                batched_discrepancies = L2discrepancy(y.detach())
            elif args.loss_fn == 'approx_hickernell':
                ## compute sum over all projections with emphasized dimensionality:
                batched_discrepancies = hickernell_all_emphasized(y.detach(),args.dim_emphasize)
            else:
                print('Loss function not implemented')
            min_discrepancy, mean_discrepancy = torch.min(batched_discrepancies).item(), torch.mean(batched_discrepancies).item

            if min_discrepancy < best_loss:
                best_loss = min_discrepancy
                f = open('results/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'.txt', 'a')
                f.write(str(best_loss) + '\n')
                f.close()

                ## save MPMC points:
                PATH = 'outputs/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'.npy'
                y = y.detach().cpu().numpy()
                np.save(PATH,y)

                torch.save(model.state_dict(), 'model_state.pth')

            if (min_discrepancy > best_loss and (epoch + 1) >= start_reduce):
                patience += 1

            if (epoch + 1) >= start_reduce and patience == reduce_point:
                patience = 0
                args.lr /= 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

            if (args.lr < 1e-6):
                f = open('results/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'.txt', 'a')
                f.write('### epochs: '+str(epoch) + '\n')
                f.close()
                break
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='number of samples')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of GNN nlayers')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight_decay')
    parser.add_argument('--nhid', type=int, default=64, # speed up training
                        help='number of hidden features of GNN')
    parser.add_argument('--nbatch', type=int, default=32,
                        help='number of point sets in batch')
    parser.add_argument('--epochs', type=int, default=20000,
                        help='number of epochs')
    parser.add_argument('--start_reduce', type=int, default=10000,
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
    parser.add_argument('--dropout', type=int, default=0.2,
                        help='dropout rate.')                       

    args = parser.parse_args()
    train(args)


