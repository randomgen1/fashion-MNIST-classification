'''
The code contains utility functions/methods to be used while training, like data transform
'''
import torch
from torchvision import transforms

###########################################################################
# Data transforms
###########################################################################
data_transforms = {
    'val': transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,),)
                               ]),

    'train_aug': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),  # data aug
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # data aug
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,),)
    ]),
    'train_transfer_learning': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_transfer_learning': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }

###########################################################################
# Used to save the best model during training
###########################################################################
class ModelCheckpoint:

    def __init__(self, filepath, state):
        self.min_loss = None
        self.filepath = filepath
        self.state = state

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.state, self.filepath)
            # torch.save(self.model.state_dict(), self.filepath)
            #torch.save(self.model, self.filepath)
            self.min_loss = loss

###########################################################################

class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)

###########################################################################
