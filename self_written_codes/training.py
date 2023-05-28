import torch.nn as nn
import torch.optim as optim

from torch.utils.data import  DataLoader
from torchvision import models, transforms

from dataset import TinyImageNet


# define a function that preporcesses the dataset
def PreprocessDataset(root_dir, BATCH_SIZE):
    # prepare Tiny-ImageNet dataset and preprocess it
    transform_train = transforms.Compose([
        # data augmentation
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        # whiten the image
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        # add gaussian noise from scratch
        
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
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
    
    # use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=5e-4)
    return net, criterion, optimizer

# write a function that adjusts the learning rate
def AdjustLearningRate(optimizer, epoch, initial_lr):
    lr = initial_lr
    if epoch >= 80:
        lr /= 10
    if epoch >= 90:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        