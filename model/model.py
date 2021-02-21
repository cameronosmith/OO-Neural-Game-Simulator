# Creates the model
# Model is composed of: 
# res dense net for image feature extraction -> 
# graph network for object groupings ->
# spatial broadcast decoder for image reconstruction and
# graph network for motion prediction

# Actually, instead of spatial broadcast decoder, implicit neural renderer which takes in a node n,
# collect all sub-nodes at the first layer to find the leftmost, rightmost, topmost, bottommost pixels 
# defining the rectangle for which the object describes, then we use a hypernetwork which takes the top level 
# node's feature vector and outputs an mlp which describes the occupancy at each X,Y,Z value and a corresponding 
# rgb vector (four dimensional output) where Z ranges from [1,2,3] and X,Y range from the rectangle's boundaries.

import torchvision.models as models
import torch.nn as nn
import torch

import model.rdn as rdn

class model(nn.Module):
    
    # Creates the sum-modules
    # For now just hard-code hyperparameters but later use config
    def __init__(self):
        print("hi")

        super(model, self).__init__()

        # Image feature extraction RDN
        self.rdn_encoder = rdn.RDN(num_channels=3, num_features=16, growth_rate=16, num_blocks=8, num_layers=5).to(device)
        
        # Graph network with hierarchical pooling
        
        # Image reconstruction spatial broadcast decoder
        
        # Graph network for dynamics prediction
        
        
    # Forward pass through model
    # img_pairs : two consecutive frames
    def forward(self,img_pairs):
        pass
        
 