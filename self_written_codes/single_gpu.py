# This is a program that trains a ResNet-18 model on Tiny-ImageNet dataset using a single GPU

# import necessary libraries
import torch 
import argparse

from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime

# import private libraries
from training import PreprocessDataset, PrepareNetwork, AdjustLearningRate

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
parser.add_argument('--root_dir', default='/data/bitahub/Tiny-ImageNet', type=str, help='root directory')
parser.add_argument('--train_dir', default='/data/bitahub/Tiny-ImageNet/train', type=str, help='train directory')
parser.add_argument('--test_dir', default='/data/bitahub/Tiny-ImageNet/test', type=str, help='test directory')
parser.add_argument('--val_dir', default='/data/bitahub/Tiny-ImageNet/val', type=str, help='val directory')
parser.add_argument('--output', default='/output', help='folder to output images and model checkpoints') 
args = parser.parse_args()

# set hyperparameters
EPOCH = 100
pre_epoch = 0
BATCH_SIZE = 64
LR = 0.0001

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