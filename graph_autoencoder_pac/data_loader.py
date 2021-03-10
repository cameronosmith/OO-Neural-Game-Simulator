# Loads the dataset
# Each data item is a consecutive pair of frames
# Valid frames are 472-5000
# Eventually split into train/test

from PIL   import Image
from torch import tensor
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms, utils

class PacmanDataset(Dataset):

    def __init__(self,imsize):
        # Generate consecutive frame paths for every frame starting after loading screen frame
        self.data_paths = ["frames/pacman_%d.jpg"%i for i in range(480,5500)]
        self.transform  = transforms.Compose([transforms.Resize((imsize,imsize)),
                                                        transforms.ToTensor()])
    # Loads a pair of images
    def __getitem__(self, index):
        return self.transform(Image.open(self.data_paths[index]))

    def __len__(self):
        return len(self.data_paths)

# Constructs train and test dataloaders
def get_datasets(imsize,batch_size):
    dataset  = PacmanDataset(imsize)
    return DataLoader(dataset,shuffle=True,pin_memory=True,batch_size=batch_size)
    #datasets = random_split(dataset,[int(len(dataset)*.9),int(len(dataset)*.1)])
    #return [DataLoader(d,shuffle=True,pin_memory=True,batch_size=batch_size) for d in datasets]
