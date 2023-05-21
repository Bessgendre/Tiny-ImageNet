import torch.nn as nn
import torch.optim as optim

from torch.utils.data import  DataLoader
from torchvision import models, transforms

from dataset import TinyImageNet


# define a function that preporcesses the dataset
def PreprocessDataset(root_dir, BATCH_SIZE):
    # prepare Tiny-ImageNet dataset and preprocess it
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    ])
    
    trainset = TinyImageNet(root_dir, transform=transform_train, train=True)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    testset = TinyImageNet(root_dir, transform=transform_test, train=False)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return trainloader, testloader

# write a function that prepares the network and do all things before training
def PrepareNetwork(initial_lr):
    net = models.resnet18(pretrained=False, num_classes=200)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    return net, criterion, optimizer

# define a function that can adjust learning rate
def AdjustLearningRate(optimizer, epoch, initial_lr):
    lr = initial_lr
    if epoch > 90:
        lr /= 100000
    if epoch > 75:
        lr /= 50000
    if epoch > 60:
        lr /= 12500
    elif epoch > 45:
        lr /= 4000
    elif epoch > 30:
        lr /= 1000
    elif epoch > 15:
        lr /= 100
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr