# This is a program that trains a ResNet-18 model on Tiny-ImageNet dataset


# import necessary libraries
import torch 
import torchvision

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms

import argparse

from torch.utils.tensorboard import SummaryWriter

import datetime
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
parser.add_argument('--train_dir', default='tiny-imagenet-200/train', type=str, help='train directory')
parser.add_argument('--test_dir', default='tiny-imagenet-200/val', type=str, help='test directory')
args = parser.parse_args()

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

trainset = datasets.ImageFolder(root=args.train_dir, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset = datasets.ImageFolder(root=args.test_dir, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# import a ResNet-18 model
net = models.resnet18(pretrained=False, num_classes=200)
print(net)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# write a function to train the model
# def train(epoch):
#     print('\nEpoch: %d' % (epoch + 1))
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
    
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs = inputs.to(device)
#         targets = targets.to(device)
        
#         optimizer.zero_grad()
        
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()
        
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
        
#         # print loss every 50 batches
#         if batch_idx % 50 == 0:
#             print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        
# write a function to test the model
# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs = inputs.to(device)
#             targets = targets.to(device)
            
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
            
#             test_loss += loss.item()
            
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
            
#             # print loss every 50 batches
#             if batch_idx % 50 == 0:
#                 print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
# train the model
if __name__ == '__main__':
    print("Start Training...")
    #生成图模型样例
    dummy_input = torch.rand(4, 3, 224, 224)
    
    #指定 Tensorboard 存储路径  文件夹名称：时间_网络结构名称
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        '/output', 'logs', current_time + '_' + "resnet18")
    
    #生成 Tensorboard writer，指定存储路径
    with SummaryWriter(logdir) as writer:
        #将模型写入 Tensorboard
        writer.add_graph(net, (dummy_input,))
        net.to(device)
        for epoch in range(pre_epoch, pre_epoch + EPOCH):
            
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
                
                # print loss every 50 batches
                if i % 50 == 0:
                    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(i+1), 100.*correct/total, correct, total))
                    #将 loss 写入 Tensorboard
                    writer.add_scalar('Train/Loss', train_loss/(i+1), epoch)
                    #将 accuracy 写入 Tensorboard
                    writer.add_scalar('Train/Accuracy', 100.*correct/total, epoch)
            
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
                
                print('Test Accuracy of the model on the test images: %.3f%% (%d/%d)' % (100 * correct / total, correct, total))
                #将 test accuracy 写入 Tensorboard
                writer.add_scalar('Test/Accuracy', 100 * correct / total, epoch)
                
                # save the best model
                acc = 100. * correct / total
                if acc > best_acc:
                    print("Saving Best Model...")
                    torch.save(net.state_dict(), '/output/resnet18.pth')
                    best_acc = acc
                    
        print("Training Finished, TotalEPOCH=%d" % EPOCH)
                
                
        
    
    



    
# save the model
torch.save(net.state_dict(), ' /output/resnet18.pth')