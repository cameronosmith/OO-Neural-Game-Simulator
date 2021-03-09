# Creates the model
# Model is composed of: 
# res dense net for image feature extraction -> 
# graph network for object groupings ->
# spatial broadcast decoder for image reconstruction and
# graph network for motion prediction

# Actually, instead of spatial broadcast decoder, last layer is some
# aggregation function which takes in a set of nodes and writes to xy locations
# of an image (or a hypernetwork?) as in seq2seq paper
#
# notes: the aggregation function is defined in torch geom as set2set pooling 
# can we go have set2set type function output weights instead of going through
# vector? i don't think so. set2set produces vector and then we use hypernetwork
# below to generate image.
# actually maybe the neural network can just be one iteration of query vectors
# and output the network
# hypernetwork here: https://github.com/vsitzmann/siren/blob/master/meta_modules.py
# and see its usage in the conv part
#
# and for graph dynamics prediction, just use DNAConv on top layer to generate
# each timestep. looks like a good conv operator to use at every layer anyway
# but top level's conv is just for through time prediction

import torchvision.models as models
import torch.nn as nn
import torch

from torch_geometric.data import Data,Batch

from numpy import arange

from model import conv_feature_extractor
from model import graph_bottleneck
from model import hypernet_img_gen

class model(nn.Module):

    # Creates the sum-modules
    # For now just hard-code hyperparameters but later use config
    def __init__(self,imsize,batch_size):

        super(model, self).__init__()
        
        # RDN image feature extraction
        feat_size=8
        self.conv_encoder = conv_feature_extractor.RDN(num_channels=3,num_features=feat_size-2,
                                   growth_rate=feat_size-2,num_blocks=4,num_layers=3)
        
        ### Define constant spatial grid (two feature maps and an edge tensor for grid linkage)
        
        grids = torch.cat([x.unsqueeze(0) for x in torch.meshgrid([torch.linspace(-1, 1, imsize) 
                                                                            for _ in range(2)])])
        self.batch_coords = grids.unsqueeze(0).repeat(batch_size,*[1]*len(grids.shape)).cuda()
        
        # Maps x,y to index in flattened list
        flat_indices = arange(imsize**2).reshape((imsize,imsize)) 
        # Create flattened adjacency list
        adj_list = [[flat_indices[x,y],flat_indices[neighbx,neighby]]
                        # For (very cell
                        for x,y in [(x,y) for x in range(imsize) for y in range(imsize)]
                            # Every valid neighbor of current cell
                            for neighbx,neighby in [(x+xd,y+yd) for xd in range(-1,2) for yd in range(-1,2) 
                                           if -1<x+xd<imsize and -1<y+yd<imsize] ]
        self.spatial_edges = torch.tensor(adj_list).t().contiguous()
        
        # Graph network with hierarchical pooling
        self.graph_encoder = graph_bottleneck.ObjGraph(feat_size,3).cuda()

        # Image reconstruction via hypernetwork
        self.img_gen       = hypernet_img_gen.HyperNetImgGen(imsize,feat_size,feat_size,batch_size).cuda()
        
    # Forward pass through model
    # img_pairs : two consecutive frames
    def forward(self,img):
        
        # Extract feature map and add spatial coordinates
        
        feats        = self.conv_encoder(img.cuda())
        feats_coords = torch.cat([self.batch_coords,feats],axis=1)
        
        flat_feats   = torch.flatten(feats_coords,2,3).permute(0,2,1)
        batched_feats= Batch.from_data_list([Data(x.cuda(),self.spatial_edges.cuda()) for x in flat_feats])
        
        # To graph representation (curr frame and next frame pred graph)
        out_graphs   = self.graph_encoder(batched_feats)
        
        # To generated images of curr frame and next frame
        curr_recon,next_recon = [self.img_gen(graph) for graph in out_graphs]
        
        return curr_recon,next_recon
        
        
