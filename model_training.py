'''
Written by Sonali Andani, for questions contact "andani.son@gmail.com
The code is used for model training. The model configuration, parameters need to be set in "config.py" file

'''

import time
import torch
import matplotlib.pyplot as plt
import os
from torchvision import datasets
import torchvision
from torch import nn, optim
from torchsummary import summary
import models
from torch.utils.tensorboard import SummaryWriter
from utils import ModelCheckpoint, data_transforms, DatasetTransformer
import csv
import config

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
tensorboard_writer = SummaryWriter(log_dir = config.logdir_visualise)

###########################################################################
# Data download and pre-processing
###########################################################################
n_classes = config.n_classes
val_ratio = 0.2 # 20 percent of data to be used for validation
batch_size = config.batch_size

# check if gpu available
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')
    n_threads = 0
else:
    device = torch.device('cpu')
    n_threads = 4

train_valid_dataset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = None)

# Split it into training and validation
n_train = int((1.0 - val_ratio) * len(train_valid_dataset))
n_val = int(val_ratio * len(train_valid_dataset))
print(n_train, n_val)

train_dataset, val_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [n_train, n_val])

if(config.model=='resnet18_pretrained'):
    train_dataset = DatasetTransformer(train_dataset, data_transforms['train_transfer_learning'])
    val_dataset = DatasetTransformer(val_dataset, data_transforms['val_transfer_learning'])
else:
    train_dataset = DatasetTransformer(train_dataset, data_transforms['train_aug'])
    val_dataset = DatasetTransformer(val_dataset, data_transforms['val'])


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = batch_size, shuffle = True, num_workers=n_threads)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size = batch_size, shuffle = False, num_workers=n_threads)

print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
print("The validation set contains {} images, in {} batches".format(len(val_loader.dataset), len(val_loader)))

###########################################################################
# Define the network
###########################################################################

if(config.model=='LeNet_like_CNN'):
    model = models.LeNet_like_CNN(num_classes=n_classes)

elif(config.model=='VGG_like_CNN'):
    model = models.VGG_like_CNN(num_classes=n_classes)

elif(config.model=='resnet10'):
    model = models.resnet10(num_classes=n_classes)

elif(config.model=='resnet18_pretrained'):
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)

model.to(device)
# summary(model, input_size=(1, 28,28))

criterion = config.criterion
optimizer = optim.Adam(model.parameters(), lr = config.learning_rate) # optimiser
epochs = config.epochs

train_losses, val_losses = [], []
val_accuracy = []

###########################################################################
# Model training
###########################################################################

for e in range(epochs):
    start_epoch = time.time()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Training pass
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        val_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computation
        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()

            # Validation pass
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                log_ps = model(images)
                val_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train()
        train_loss_epoch = running_loss / len(train_loader)
        val_loss_epoch = val_loss / len(val_loader)
        val_accuracy_epoch = accuracy / len(val_loader)

        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)
        val_accuracy.append(val_accuracy_epoch)

        print("Epoch: {}/{}..".format(e + 1, epochs),
              "Training loss: {:.3f}..".format(train_loss_epoch),
              "Val loss: {:.3f}..".format(val_loss_epoch),
              "Val Accuracy: {:.3f}".format(val_accuracy_epoch))

        tensorboard_writer.add_scalar('metrics/train_loss', train_loss_epoch, e)
        tensorboard_writer.add_scalar('metrics/val_loss', val_loss_epoch, e)
        tensorboard_writer.add_scalar('metrics/val_acc', val_accuracy_epoch, e)

        # saving best model
        state = {
            'epoch': e+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        filepath = os.path.join(config.logdir, "best_model.pt")
        model_checkpoint = ModelCheckpoint(filepath, state)
        model_checkpoint.update(val_loss_epoch)

        end_epoch = time.time()
        print('time for epoch: ', (end_epoch - start_epoch))
        print('\n')

        # saving model after every 5 iterations
        if(e%5 == 0):
            file_name = "model_" + str(e+1) + '-' + "{:.3f}".format(val_loss_epoch) \
                        + '-' + "{:.3f}".format(val_accuracy_epoch) + '.pt'
            filepath = os.path.join(config.logdir, file_name)
            torch.save(state, filepath)

plt.plot(train_losses, label = "Training loss")
plt.plot(val_losses, label = "Validation loss")
plt.legend(frameon = False)
plt.savefig(os.path.join(config.logdir, 'plot.png'))
plt.show()

# writing the metrics in case we need to plot it later
f = open(os.path.join(config.logdir, 'train_output.csv'),'a')
wr = csv.writer(f, dialect='excel')
wr.writerow(train_losses)
wr.writerow(val_accuracy)
wr.writerow(val_losses)
f.close()
#####################################################################################
