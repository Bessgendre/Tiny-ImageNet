# This is a program that trains a ResNet-18 model on Tiny-ImageNet dataset


# import necessary libraries
import torch 
import torchvision

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms

# set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# set hyperparameters
EPOCH = 3
pre_epoch = 0
BATCH_SIZE = 64
LR = 0.01

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

trainset = datasets.ImageFolder(root='/Users/royalty/Desktop/Python_ML/Python-DL/second_assignment/tiny-imagenet-200/train', transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = datasets.ImageFolder(root='/Users/royalty/Desktop/Python_ML/Python-DL/second_assignment/tiny-imagenet-200/val', transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# import a ResNet-18 model
net = models.resnet18(pretrained=False, num_classes=200)
net = net.to(device)
print(net)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# write a function to train the model
def train(epoch):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # print loss every 50 batches
        if batch_idx % 50 == 0:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        
# write a function to test the model
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # print loss every 50 batches
            if batch_idx % 50 == 0:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
# train the model
for epoch in range(pre_epoch, pre_epoch + EPOCH):
    train(epoch)
    test(epoch)
    
# save the model
torch.save(net.state_dict(), 'resnet18.pth')