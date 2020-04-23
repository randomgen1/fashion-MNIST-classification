import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torchvision import datasets
import models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import config
from utils import data_transforms, DatasetTransformer
from torch import nn
import torchvision

###########################################################################
# Data download and pre-processing
###########################################################################
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')
    n_threads = 0
else:
    device = torch.device('cpu')
    n_threads = 4

test_dataset = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = None)

if(config.model=='resnet18_pretrained'):
    test_dataset = DatasetTransformer(test_dataset, data_transforms['val_transfer_learning'])
else:
    test_dataset = DatasetTransformer(test_dataset, data_transforms['val'])

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = 100, shuffle = False, num_workers=n_threads)

print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))

###########################################################################
# Loading the network
###########################################################################
if(config.model=='LeNet_like_CNN'):
    model = models.LeNet_like_CNN(num_classes=config.n_classes)

elif(config.model=='VGG_like_CNN'):
    model = models.VGG_like_CNN(num_classes=config.n_classes)

elif(config.model=='resnet10'):
    model = models.resnet10(num_classes=config.n_classes)

elif(config.model=='resnet18_pretrained'):
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.n_classes)

# Loading model weights
model.to(device)
filepath = os.path.join(config.logdir, "best_model.pt")
state = torch.load(filepath, map_location=lambda storage, loc: storage)
model.load_state_dict(state['state_dict'])
model.eval()
criterion = torch.nn.CrossEntropyLoss()
print('model loaded')

loss = 0
accuracy = 0
y_pred = []
y_true = []

with torch.no_grad():
    model.eval()

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        log_ps = model(images)
        loss += criterion(log_ps, labels)

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        y_pred.append(top_class.cpu().numpy())
        y_true.append(labels.cpu().numpy())

        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

    test_loss = loss / len(test_loader)
    test_accuracy = accuracy / len(test_loader)

print('test_loss: ', test_loss)
print('test_accuracy: ', test_accuracy)

y_pred = np.array(y_pred).flatten()
y_true = np.array(y_true).flatten()

#####################################################################################
# plotting and saving metrics -- confusion matrix, precision, recall, accuracy
#####################################################################################

cm = confusion_matrix(y_true, y_pred)
print('confusion_matrix: ',cm)

# confusion matrix plot
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'bag', 'Ankle boot']
plt.title('Confusion matrix ')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)

plt.colorbar()
plt.savefig(os.path.join(config.logdir, 'confusion_matrix.png'))
plt.show()

# precision, recall, F1 score
p_r_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')
print('Precision, recall, F1 score: ', p_r_f1)

# saving results in the file
with open(os.path.join(config.logdir, 'test_output.txt'), 'a') as f:
    f.write('test precision, recall, F1: ' + str(p_r_f1) + os.linesep)
    f.write('test accuracy : ' + str(test_accuracy) + os.linesep)
    f.write('test loss : ' + str(test_loss) + os.linesep)
    f.write('confusion matrix : ' + str(cm) + os.linesep)

#####################################################################################
