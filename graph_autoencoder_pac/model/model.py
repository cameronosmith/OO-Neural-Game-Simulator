# RDN feature extractor maps image pixel to feature
# Convert to graph (just plain grid)
# Decoder MLP takes maps feature vector from graph to pixel

import torchvision.models as models
import torch.nn as nn
import torch

from torch_geometric.data  import Data,Batch
from torch_geometric.utils import to_dense_batch

from numpy import arange

from model import conv_feature_extractor
from model.vsitzmann_submodules import vsitzmann_pytorch_prototyping as pytorch_prototyping

class model(nn.Module):

    # Creates the sum-modules
    # For now just hard-code hyperparameters but later use config
    def __init__(self,imsize,batch_size):

        super(model, self).__init__()
        
        # RDN image feature extraction
        feat_size=8
        self.conv_encoder = conv_feature_extractor.RDN(num_channels=3,num_features=feat_size-2,
                                   growth_rate=feat_size-2,num_blocks=4,num_layers=3).cuda()
        
        ## Define constant spatial grid (two feature maps and an edge tensor for grid linkage)
        
        grids = torch.cat([x.unsqueeze(0) for x in torch.meshgrid([torch.linspace(-1, 1, imsize) 
                                                                   for _ in range(2)])])
        self.batch_coords = grids.unsqueeze(0).repeat(batch_size,*[1]*len(grids.shape)).cuda()
        
        ## Create constant grid edge connnections for graph
        # Maps x,y to index in flattened list
        flat_indices = arange(imsize**2).reshape((imsize,imsize)) 
        # Create flattened adjacency list
        adj_list = [[flat_indices[x,y],flat_indices[neighbx,neighby]]
                        # For (very cell
                        for x,y in [(x,y) for x in range(imsize) for y in range(imsize)]
                            # Every valid neighbor of current cell
                            for neighbx,neighby in [(x+xd,y+yd) 
                                                for xd in range(-1,2) for yd in range(-1,2) 
                                           if -1<x+xd<imsize and -1<y+yd<imsize] ]
        self.spatial_edges = torch.tensor(adj_list).t().contiguous().cuda()
        
        # Graph network with hierarchical pooling TODO once autoencoder verified
        # self.graph_encoder = graph_bottleneck.ObjGraph(feat_size,3).cuda()
        
        # MLP that maps feature vector to pixel

        self.pixel_generator = pytorch_prototyping.FCBlock(hidden_ch=64,
                                                           num_hidden_layers=3,
                                                           in_features=feat_size,
                                                           out_features=3,
                                                           outermost_linear=True).cuda()

    # Forward pass through model
    # img_pairs : two consecutive frames
    def forward(self,img):
        
        # Extract feature map and add spatial coordinates
        
        feats        = self.conv_encoder(img.cuda())
        feats_coords = torch.cat([self.batch_coords,feats],axis=1).cuda()
        flat_feats   = feats_coords.flatten(2,3).permute(0,2,1).cuda()
        graph_in     = Batch.from_data_list([Data(x,self.spatial_edges ) for x in flat_feats])
        
        # To graph representation (curr frame and next frame pred graph)
        # TODO after verified autoencoder works, for now just convert back
        # out_graphs   = self.graph_encoder(batched_feats) 
        graph_out = to_dense_batch(graph_in.x,graph_in.batch)[0].cuda()
        
        # Map each image coordinate to a pixel, where each feature vec in the graph
        # contributes an amount proportional to how close its first two (spatial) 
        # dimensions match the image coordinate
        crds   = self.batch_coords.flatten(2,3).permute(0,2,1).cuda().contiguous()
        graph_spatial_feats = graph_out[:,:,:2].contiguous()
        score  = 1/(1+torch.cdist(graph_spatial_feats,crds,compute_mode="use_mm_for_euclid_dist").permute(0,1,2)).cuda()
        soft_score   = torch.nn.Softmax(dim=1)(20*score).cuda()
        weight_feats = torch.matmul(soft_score,graph_out).cuda() # B x imsize**2 * feats
        
        return self.pixel_generator(weight_feats).cuda()
    
        
    
   
    
    