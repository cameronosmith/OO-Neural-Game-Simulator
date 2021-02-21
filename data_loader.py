# Loads the dataset
# Each data item is a consecutive pair of frames
# Valid frames are 472-5000
# Eventually split into train/test

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class PacmanDataset(Dataset):
    
    def __init__(self):
        # Generate consecutive frame paths for every frame starting after loading screen frame
        self.data_paths = [["frames/pacman_%d.jpg"%j for j in range(2)] for i in range(472,1000)]  
        self.transform  = transforms.Compose([transforms.Resize(256)])
        
    # Loads a pair of images
    def __getitem__(self, index):
        return [self.transform(Image.open(path)) for path in self.data_paths[index]]

    def __len__(self):
        return len(self.data_paths)
    
# Just constructs the dataloader for class above
def pacman_dataloader():
    return DataLoader(PacmanDataset(),shuffle=True,pin_memory=True,batch_size=4)
