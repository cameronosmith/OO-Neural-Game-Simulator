# DynamicEdgeConv for convolution
# ASAP pooling operator
#
# DNAConv on top layer for dynamics prediction
#
# Code copied from pytorch example: 
# github.com/rusty1s/pytorch_geometric/

from torch.nn              import Module, ModuleList
from torch.nn.functional   import relu
from torch_geometric.nn    import GraphConv, EdgePooling
from torch_geometric.nn    import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_batch
from torch_geometric.data  import Data

class ObjGraph(Module):
  
    def __init__(self, hidden_dim, num_layers):
        super(ObjGraph, self).__init__()
        
        self.convs = ModuleList([GraphConv(hidden_dim,hidden_dim,aggr="max") for _ in range(num_layers)])
        self.pools = ModuleList([EdgePooling(hidden_dim) for _ in range(num_layers)])

        # Dynamics module (predicts a vector with global attention and then graph conv conditioned on it)
        # ...

    # Returns a graph vec for the input graph and also the prediction of the graph's next state ...
    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, pool in zip(self.convs,self.pools):
            x = relu(conv(x, edge_index))
            x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
            
        return to_dense_batch(x,batch)[0] # put back into BxNxF