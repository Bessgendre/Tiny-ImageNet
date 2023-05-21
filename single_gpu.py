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

import os
from datetime import datetime

import sys
from PIL import Image
    
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
LR = 0.01

# define a class that read the Tiny-ImageNet dataset
class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


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

# define a function that trains the network with tensorboard visualization
def TrainingNetworkWithTensorboard(net, criterion, optimizer, trainloader, testloader):
    net = net.to(device)
    
    #指定 Tensorboard 存储路径  文件夹名称：时间_网络结构名称
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    logdir = os.path.join(
        '/output', 'logs', current_time + '_' + "resnet18")
    
    #创建一个 SummaryWriter 对象
    writer = SummaryWriter(logdir)
    
    for epoch in range(pre_epoch, pre_epoch + EPOCH):
        adjust_learning_rate(optimizer, epoch)
        # write learning rate to tensorboard
        writer.add_scalar('training/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
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
                print('Saving model...')
                torch.save(net.state_dict(), '%s/final_net.pth' % (args.output))
                flag = 0
                
            print('Test Accuracy: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
        
        # add loss and accuracy to tensorboard
        writer.add_scalar('/testing/test_accuracy', 100.*correct/total, epoch)
        
        # add model parameters to tensorboard
        for name, param in net.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        
        # compare current accuracy with previous best accuracy, save the better one
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving model...')
            torch.save(net.state_dict(), '%s/final_net.pth' % (args.output))
            best_acc = acc
            flag = 0
                
        if epoch > 30 and flag > 7:
            break
        else:
            flag += 1
    return epoch, best_acc
            


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
                
            if epoch > 10 and flag > 7:
                break
            else:
                flag += 1
    return epoch, best_acc
        
# main fuction that when called, will start all those process
def main():
    print("Preprocessing Dataset...")
    trainloader, testloader = PreprocessDataset(args.root_dir, BATCH_SIZE)
    
    print("Preparing Network...")
    net, criterion, optimizer = PrepareNetwork()
    
    print("Start Training...")
    # real_epoch, best_accuracy = TrainingNetwork(net, criterion, optimizer, trainloader, testloader)
    real_epoch, best_accuracy = TrainingNetworkWithTensorboard(net, criterion, optimizer, trainloader, testloader)
    
    print("Training Finished!")
    print("Total epochs: %d" % (real_epoch + 1))
    print("Planed epochs: %d" % (EPOCH))
    print("Best Accuracy: %.3f%%" % (best_accuracy))

if __name__ == '__main__':
    main()