# File to create an experiment, later use config files for the hyperparameters

from os.path     import exists

import torch
from torchvision.utils import save_image

from data_loader import get_datasets
from model       import model

# Runs an experiment 
class Experiment(object):

    # Constructs the model to be run
    def __init__(self,model_file="experiments/model_check.pt"):
        
        # Use a config file later
        self.batch_size = 1
        self.imsize     = 32
        self.model_file = model_file
        
        # Load in frames dataset
        self.dataset = get_datasets(self.imsize,self.batch_size)
        
        # Create the model
        self.model      = model.model(self.imsize,self.batch_size).cuda()
        self.criterion  = torch.nn.MSELoss()
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Reload model if file exists
        if False:#exists(model_file):
            print("Loading model from file")
            checkpoint = torch.load(model_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       
    # Trains model
    def run(self):
        
        # Train for number of epochs
        for epoch in range(2000):
            print("Epoch %d"%epoch)
            # Training and then Validation on datasets
            for i,imgs in enumerate(self.dataset):
                if imgs.size(0)!=self.batch_size:continue # skip incomplete batch
                imgs=imgs.cuda()
                    
                self.optimizer.zero_grad()

                recon = self.model(imgs)
                loss  = self.criterion(recon.flatten(),imgs.flatten())
                loss.backward()
                self.optimizer.step()

                # Every 100 elements write img to file
                if i%100==0: 
                    print("epoch: %d, element: %d/%d, loss: %f"%
                               (epoch,i,len(self.dataset),loss))
                    save_image(recon[0].resize(3,self.imsize,self.imsize),
                               'experiments/output_imgs/%d_%d.png'%(epoch,i))

            # Done with epoch, save model and update stats
            self.train_val_scores.append(epoch_scores)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.model_file)

if __name__ == "__main__": Experiment().run()