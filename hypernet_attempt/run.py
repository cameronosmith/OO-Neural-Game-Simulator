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
        self.batch_size = 4
        self.imsize     = 64
        self.model_file = model_file
        
        # Load in frames dataset
        self.train_data,self.val_data = get_datasets(self.imsize,self.batch_size)
        
        # Create the model
        self.model = model.model(self.imsize,self.batch_size).cuda()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        # Records of train/val scores
        self.train_val_scores = [[],[]]
        
        # Reload model if file exists
        if exists(model_file):
            print("Loading model from file")
            checkpoint = torch.load(model_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_val_scores = checkpoint['train_val_scores']
       
    # Trains model
    def run(self):
        
        # Train for number of epochs
        for epoch in range(2000):
            print("Epoch %d"%epoch)
            epoch_scores = [0,0]
            # Training and then Validation on datasets
            for phase,dataset in enumerate([self.train_data]):
                for i,(img_t,img_t1) in enumerate(dataset):
                    if img_t.size(0)!=self.batch_size:continue # skip incomplete batch
                    img_t,img_t1 = [x.cuda() for x in [img_t,img_t1]]
                   
                    self.optimizer.zero_grad()
                    
                    # Generate imgt, imgt1, and penalty is their reconstruction 
                    # error plus the graph model's dynamics prediction difference
                    recon_t, recon_t1 = self.model(img_t)
                    loss = self.criterion(recon_t.flatten(),img_t.flatten())+\
                            self.criterion(recon_t1.flatten(),img_t1.flatten())
                            #+ self.model.graph_encoder.pred_size_penalty.flatten())
                    loss.backward()
                    self.optimizer.step()
                    
                    # Every 100 elements write img to file
                    if i%100==0: 
                        print("epoch: %d, val: %d/, element: %d/%d, loss: %f"%
                                   (epoch,phase,i,len(dataset),loss))
                        save_image(recon_t[0].resize(3,self.imsize,self.imsize),
                                   'experiments/output_imgs/%d_%d.png'%(epoch,i))
                    
                    epoch_scores[phase] += loss
                
            # Done with epoch, save model and update stats
            self.train_val_scores.append(epoch_scores)
            torch.save({
                'train_val_scores': self.train_val_scores,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.model_file)

if __name__ == "__main__": Experiment().run()