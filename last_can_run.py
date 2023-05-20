# This is a program that trains a ResNet-18 model on Tiny-ImageNet dataset


# import necessary libraries
import torch 
import torchvision

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms

import argparse

# import tensorboard
# from torch.utils.tensorboard import SummaryWriter

import os
    
# set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# set parameters
parser = argparse.ArgumentParser(description='PyTorch Tiny-ImageNet Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epoch', default=3, type=int, help='number of epochs tp train for')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--pre_epoch', default=0, type=int, help='number of epochs before training')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--train_dir', default='/data/bitahub/Tiny-ImageNet/train', type=str, help='train directory')
parser.add_argument('--test_dir', default='/data/bitahub/Tiny-ImageNet/test', type=str, help='test directory')
parser.add_argument('--output', default='/output', help='folder to output images and model checkpoints') 
args = parser.parse_args()

# set hyperparameters
EPOCH = 100
pre_epoch = 0
BATCH_SIZE = 64
LR = 0.01

# define a function that preporcesses the dataset
def PreprocessDataset(train_dir, test_dir):
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

    trainset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    testset = datasets.ImageFolder(root=test_dir, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return trainloader, testloader

# write a function that prepares the network and do all things before training
def PrepareNetwork():
    net = models.resnet18(pretrained=False, num_classes=200)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    return net, criterion, optimizer

# define a function that can adjust learning rate
def adjust_learning_rate(optimizer, epoch):
    lr = LR
    if epoch > 80:
        lr /= 100000
    elif epoch > 60:
        lr /= 10000
    elif epoch > 40:
        lr /= 1000
    elif epoch > 20:
        lr /= 100
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# define a function that trains the network
def TrainingNetwork(net, criterion, optimizer, trainloader, testloader):
    net = net.to(device)
    for epoch in range(pre_epoch, pre_epoch + EPOCH):
        adjust_learning_rate(optimizer, epoch)
        
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
            
        for i , data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # print training loss and accuracy after each 100 mini-batches
            if i % 100 == 0:
                print('Training Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(i+1), 100.*correct/total, correct, total))
        
        # test accuracy after each epoch
        print("Waiting Test...")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # save the first epoch model
            if epoch == 0:
                best_acc = 100.*correct/total
                print('Saving model...')
                torch.save(net.state_dict(), '%s/final_net.pth' % (args.output))
                flag = 0
            
            print('Test Accuracy of the model on the test images: %.3f%% (%d/%d)' % (100 * correct / total, correct, total))
            
            
            # compare current accuracy with previous best accuracy, save the better one
            acc = 100.*correct/total
            if acc > best_acc:
                print('Saving model...')
                torch.save(net.state_dict(), '%s/final_net.pth' % (args.output))
                best_acc = acc
                flag = 0
                
            if epoch > 50 and flag > 7:
                break
            else:
                flag += 1
                
        print("Training Finished, TotalEPOCH = %d, PlanedEPOCH = %d" % epoch, EPOCH)
        
# main fuction that when called, will start all those process
def main():
    print("Preprocessing Dataset...")
    trainloader, testloader = PreprocessDataset(args.train_dir, args.test_dir)
    
    print("Preparing Network...")
    net, criterion, optimizer = PrepareNetwork()
    
    print("Start Training...")
    TrainingNetwork(net, criterion, optimizer, trainloader, testloader)


if __name__ == '__main__':
    main()