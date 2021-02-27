# Takes in the last layer of a graph network, produces an implicit function
# for the generated image.
# Code directly taken from github.com/vsitzmann/siren
# Pointnet inspired architecture for hyponet layer generation

# example at https://github.com/vsitzmann/scene-representation-networks/blob/master/srns.py

import torch
from torch import nn

from model.vsitzmann_submodules import vsitzmann_hypernet_submodules as hyperlayers
from model.vsitzmann_submodules import vsitzmann_pytorch_prototyping as pytorch_prototyping

class HyperNetImgGen(nn.Module):
    # 
    # imsize : image dimension
    def __init__(self, imsize, latent_dim, hyper_hidden_dim):
        
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_hidden_units_phi = 64
        self.phi_layers = 3  # includes the in and out layers
        self.rendering_layers = 3  # includes the in and out layers
        
        #batchsize=4 dont hardcode
                
        # Image coordinates to use in image generation
        points_xy = torch.cat([x.unsqueeze(0) for x in  torch.meshgrid([torch.linspace(-1, 1, imsize) for _ in range(2)])])
        self.batch_coords_xy = points_xy.unsqueeze(0).repeat(4,*[1]*len(points_xy.shape)).reshape(4,-1,2).cuda()
        
        self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=latent_dim,
                                             hyper_num_hidden_layers=1,
                                             hyper_hidden_ch=hyper_hidden_dim,
                                             hidden_ch=self.num_hidden_units_phi,
                                             num_hidden_layers=self.phi_layers - 2,
                                             in_ch=2,
                                             out_ch=self.num_hidden_units_phi)
        
        self.pixel_generator = pytorch_prototyping.FCBlock(hidden_ch=self.num_hidden_units_phi,
                                                           num_hidden_layers=self.rendering_layers - 1,
                                                           in_features=self.num_hidden_units_phi,
                                                           out_features=3,
                                                           outermost_linear=True)


    # z      : latent vector to condition network on
    # return : generated img
    def forward(self, z):

        phi = self.hyper_phi(z) # Forward pass through hypernetwork yields a (callable) SRN.

        # Map img coordinates to feature vecs then to RGB vecs 
        v = phi(self.batch_coords_xy)
        img_out = self.pixel_generator(v)

        return img_out
