import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

from nets import alexnet
from nets import simplenet


def estimate(net, dataloader, device):
    label_true = 0
    label_all = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device=device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.to(device='cpu').numpy()
        labels = labels.numpy()
        label_true += np.sum(predicted - labels == 0)
        label_all += len(labels)
    return label_true/label_all


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size_train = 4
    epoch_size = 1
    print('Using device %s' % device)
    # load dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)
    estimate_loader = torch.utils.data.DataLoader(
        trainset, batch_size=100, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                'ship', 'truck')
    net = alexnet.Net(len(classes))
    net.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # train
    accuracy = []
    for epoch in range(1):
        print('epoch %d...' % (epoch+1))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 2000 mini-batches
                print('[%d, %d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0
        # ac_tmp = estimate(net, estimate_loader, device)
        # print(ac_tmp)
        # accuracy.append(ac_tmp)
    print('Finished Training')
    print(accuracy)
    return net
    

def test(net):
    batch_size_test = 20
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                'ship', 'truck')
    # load dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images = images.to(device=device)
    labels = labels.to(device=device)
    print('GroundTruth: ',
        ' '.join('%5s' % classes[labels[j]] for j in range(batch_size_test)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ',
        ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size_test)))


if __name__ == '__main__':
    net = train()
    test(net)
