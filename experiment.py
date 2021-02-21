# File to create an experiment, later use config files for the hyperparameters

from data_loader import pacman_dataloader
from model       import model

# Runs an experiment 
class Experiment(object):

    # Constructs the model to be run
    def __init__(self):
        
        # Load in frames dataset
        self.frames = pacman_dataloader()
        
        # Create the model
        self.model = model.model()
    
    # Trains and tests the model
    def run(self):
        pass
        