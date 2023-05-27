# This is a program that trains a ResNet-18 model on Tiny-ImageNet dataset using more than one GPU

# import necessary libraries
import torch 
import argparse

from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime

# import private libraries
from training import PreprocessDataset, PrepareNetwork, AdjustLearningRate

# find out how many GPUs are available
ngpu = torch.cuda.device_count()
print("Number of GPUs: ", ngpu)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# set parameters
parser = argparse.ArgumentParser(description='PyTorch Tiny-ImageNet Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epoch', default=3, type=int, help='number of epochs tp train for')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--pre_epoch', default=0, type=int, help='number of epochs before training')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--root_dir', default='/data/bitahub/Tiny-ImageNet', type=str, help='root directory')
parser.add_argument('--output', default='/output', help='folder to output images and model checkpoints') 
args = parser.parse_args()

# set hyperparameters
EPOCH = 100
pre_epoch = 0
BATCH_SIZE = 64
LR = 0.0001

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import  DataLoader
from torchvision import models, transforms

from dataset import TinyImageNet

def init_distributed_mode(args):
    # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
    # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
    if'RANK'in os.environ and'WORLD_SIZE'in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        # LOCAL_RANK代表某个机器上第几块GPU
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif'SLURM_PROCID'in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)  # 对当前进程指定使用的GPU
    args.dist_backend = 'nccl'# 通信后端，nvidia GPU推荐使用NCCL
    dist.barrier()  # 等待每个GPU都运行完这个地方以后再继续

# define a function that preporcesses the dataset
def PreprocessDataset(root_dir, batch_size):
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
    # 给每个rank对应的进程分配训练的样本索引
    train_sampler=torch.utils.data.distributed.DistributedSampler(trainset)
    
    testset = TinyImageNet(root_dir, transform=transform_test, train=False)
    test_sampler=torch.utils.data.distributed.DistributedSampler(testset)
    #将样本索引每batch_size个元素组成一个list
    
    train_batch_sampler=torch.utils.data.BatchSampler(train_sampler,batch_size,drop_last=True)
    
    
    trainloader = DataLoader(trainset,
                             batch_size=batch_size, 
                             batch_sampler=train_batch_sampler,
                             shuffle=True, 
                             pin_memory=True,
                             num_workers=4)
    
    
    testloader = DataLoader(testset, 
                            batch_size=batch_size,
                            sampler=test_sampler,
                            shuffle=False, 
                            pin_memory=True,
                            num_workers=4)

    return trainloader, testloader

# write a function that prepares the network and do all things before training
def PrepareNetwork(initial_lr):
    net = models.resnet18(pretrained=False, num_classes=200)
    criterion = nn.CrossEntropyLoss()
    
    # use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=5e-4)
    return net, criterion, optimizer

# define a function that trains the network with tensorboard visualization
def TrainingNetwork(net, criterion, optimizer, trainloader, testloader):
    net = net.to(device)
    
    # set Tensorboard directory
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    logdir = os.path.join(
        '/output', 'logs', current_time + '_' + "resnet18")
    
    # create a SummaryWriter object
    writer = SummaryWriter(logdir)
    
    for epoch in range(pre_epoch, pre_epoch + EPOCH):
        # AdjustLearningRate(optimizer, epoch, LR)
        
        # write learning rate to tensorboard
        writer.add_scalar('training/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
            
        # train the network
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
        
        # write training loss and accuracy to tensorboard
        writer.add_scalar('training/train_loss', train_loss/(i+1), epoch)
        writer.add_scalar('training/train_acc', 100.*correct/total, epoch)
            
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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                
            # save the first epoch model
            if epoch == 0:
                best_acc = 100.*correct/total
                print('Saving Best Model...')
                torch.save(net.state_dict(), '%s/best_on_val.pth' % (args.output))
                
            # print testing accuracy after each epoch
            print('Test Accuracy: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
        
        # add loss and accuracy to tensorboard
        writer.add_scalar('/testing/test_accuracy', 100.*correct/total, epoch)
        
        # add model parameters to tensorboard
        for name, param in net.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        
        # compare current accuracy with previous best accuracy, save the better one
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving Best Model...')
            torch.save(net.state_dict(), '%s/best_on_val.pth' % (args.output))
            best_acc = acc
    
    torch.save(net.state_dict(), '%s/final_net.pth' % (args.output))
    return epoch, best_acc
            
# main fuction that when called, will start all those process
def main():
    print("Preprocessing Dataset...")
    trainloader, testloader = PreprocessDataset(args.root_dir, BATCH_SIZE)
    
    print("Preparing Network...")
    net, criterion, optimizer = PrepareNetwork(LR)
    
    print("Start Training...")
    real_epoch, best_accuracy = TrainingNetwork(net, criterion, optimizer, trainloader, testloader)
    
    print("Training Finished!")
    print("Total epochs: %d" % (real_epoch + 1))
    print("Planed epochs: %d" % (EPOCH))
    print("Best Accuracy: %.3f%%" % (best_accuracy))

if __name__ == '__main__':
    main()