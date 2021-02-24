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

from model import conv_feature_extractor
from model import graph_bottleneck

class model(nn.Module):

    # Creates the sum-modules
    # For now just hard-code hyperparameters but later use config
    def __init__(self, device="cpu"):

        super(model, self).__init__()

        # RDN image feature extraction
        self.conv_encoder = conv_feature_extractor.RDN(num_channels=3,
            num_features=16,growth_rate=16, num_blocks=8, num_layers=5)

        # Graph network with hierarchical pooling
        #self.graph_encoder = graph_bottleneck()

        # Image reconstruction spatial broadcast decoder
        # Graph network for dynamics prediction

    # Forward pass through model
    # img_pairs : two consecutive frames
    def forward(self,img_pairs):

        # Add spatial location to conv encoder features
        pass
