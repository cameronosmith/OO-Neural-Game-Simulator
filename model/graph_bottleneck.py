# DynamicEdgeConv for convolution
# ASAP pooling operator
#
# DNAConv on top layer for dynamics prediction
#
# Code copied from pytorch example: 
# github.com/rusty1s/pytorch_geometric/

import torch #cleanup later
from torch.nn              import Module, ModuleList
from torch.nn              import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.utils import to_dense_batch
from torch.nn.functional   import relu
from torch_geometric.nn    import SAGPooling,EdgeConv
from torch_geometric.data  import Data
from torch_scatter         import scatter_add

"""
# Pooling disturbs batch ordering so manually fixing it by
# changing the disturbed line in utils.to_dense_batch
def to_dense_batch(x,batch):
    batch_size = batch.max()+1
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0,
                            dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    max_num_nodes = num_nodes.max().item()

    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, 0)
    out[idx] = x
    return out.view([batch_size, max_num_nodes] + list(x.size())[1:])
"""

class ObjGraph(Module):
  
    def __init__(self, hidden_dim, num_layers):
        super(ObjGraph, self).__init__()
        
        # Layers used for object coarsening
        self.pools = ModuleList([SAGPooling(hidden_dim,min_score=.5) for _ in range(num_layers)])

        # Dynamics prediction module , note to penalize difference between pred and current (for small modifications)
        channels=[hidden_dim*2,hidden_dim*3,hidden_dim]
        mlp=Seq(*[ Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
                                                        for i in range(1, len(channels)) ])
        self.dynamics_conv = EdgeConv(mlp)
        # Holds the difference between pred graph state and curr graph state for additional penalty
        self.penalty_criterion = torch.nn.L1Loss()
        self.pred_size_penalty = 0

    # Returns a graph vec for the input graph and also the prediction of the graph's next state ...
    def forward(self, data):
        
        # Object coarsening
        x, edge_index, batch = data.x, data.edge_index, data.batch
      
        for pool in self.pools:
            x, edge_index, _, batch, _, _ = pool(x, edge_index, batch=batch)
            
        # Dynamics prediction
        xp = self.dynamics_conv(x,edge_index)
        self.pred_size_penalty = self.penalty_criterion(xp,x)
        
        # Pack nodes back into (BxNxF) shape
        batched_x  = to_dense_batch(x,batch)[0]
        batched_xp = to_dense_batch(xp,batch)[0]
        
        return batched_x,batched_xp