import torch
import torch.nn as nn
from d2l import torch as d2l

# write a trainer function that can do the training process on GPU

def TrainModel(net, train_iter, test_iter, loss, num_epochs, optimiser, device):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimiser.zero_grad()
            l.backward()
            optimiser.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            
        
        # print every 100 epoch
        if (epoch + 1) % 100 == 0:
            test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
            print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f" % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc))
            
# define a new dataset with pytorch Datasets class to load Tini-ImageNet dataset



class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, root, is_train=True, transform=None):
        self.transform = transform
        self.is_train = is_train
        self.label_to_idx = {}
        self.images = []
        self.labels = []
        self.classes = []
        self.root = root
        self.load()
        
    def load(self):
        if self.is_train:
            root = self.root + '/train'
        else:
            root = self.root + '/val'
        for label in os.listdir(root):
            if label not in self.label_to_idx:
                self.label_to_idx[label] = len(self.label_to_idx)
                self.classes.append(label)
            label_idx = self.label_to_idx[label]
            label_path = root + '/' + label
            for img_name in os.listdir(label_path):
                img_path = label_path + '/' + img_name
                self.images.append(img_path)
                self.labels.append(label_idx)
                
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.images)
    
