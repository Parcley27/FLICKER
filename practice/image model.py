import torch
import torchvision
from torchvision.transforms import v2

import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

import matplotlib.pyplot as plot
import numpy as np


batchSize = 4
epochCount = 5
modelPath = './practice/cifar_net.pt'
dataPath = './practice/data'

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# convert PIL images to float tensors normalized to [-1, 1]
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),  # scale pixel values from [0, 255] to [0, 1]
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # shift to [-1, 1] per channel
])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution1 = nn.Conv2d(3, 32, 5)   # 3 input channels (RGB), 32 filters, 5x5 kernel
        self.maxPool = nn.MaxPool2d(2, 2)           # halve spatial dimensions
        self.convolution2 = nn.Conv2d(32, 16, 5)   # 32 -> 16 filters, 5x5 kernel
        self.fullyConnected1 = nn.Linear(16 * 5 * 5, 120)  # flattened conv output -> 120
        self.fullyConnected2 = nn.Linear(120, 84)
        self.fullyConnected3 = nn.Linear(84, 10)   # 10 output scores, one per class

    def forward(self, x):
        x = self.maxPool(functional.relu(self.convolution1(x)))
        x = self.maxPool(functional.relu(self.convolution2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = functional.relu(self.fullyConnected1(x))
        x = functional.relu(self.fullyConnected2(x))
        x = self.fullyConnected3(x)  # raw scores (logits), no activation

        return x


def showImage(image):
    image = image / 2 + 0.5  # unnormalize
    
    numpyImage = image.numpy()

    plot.imshow(np.transpose(numpyImage, (1, 2, 0)))
    plot.show()


if __name__ == "__main__":
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')

    else:
        device = torch.device('cpu')

    # Data loading
    trainingSet = torchvision.datasets.CIFAR10(root = dataPath, train = True, download = True, transform = transform)
    trainingLoader = torch.utils.data.DataLoader(trainingSet, batch_size = batchSize, shuffle = True, num_workers = 2)

    testSet = torchvision.datasets.CIFAR10(root = dataPath, train = False, download = True, transform = transform)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, shuffle = False, num_workers = 2)

    # Preview a batch of training images
    trainingIterator = iter(trainingLoader)
    images, labels = next(trainingIterator)
    showImage(torchvision.utils.make_grid(images))
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batchSize)))

    # Network, loss, and optimizer setup
    network = Net()
    network.to(device)

    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr = 0.001, momentum = 0.9)

    # Training
    if epochCount != 0:
        for epoch in range(epochCount):
            runningLoss = 0.0

            for batchIndex, batch in enumerate(trainingLoader, 0):
                inputs, labels = batch[0].to(device), batch[1].to(device)

                optimizer.zero_grad()  # clear gradients from previous step

                # forward pass, compute loss, backward pass, update weights
                outputs = network(inputs)
                loss = lossFunction(outputs, labels)
                loss.backward()
                optimizer.step()

                runningLoss += loss.item()
                if batchIndex % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {batchIndex + 1:5d}] loss: {runningLoss / 2000:.3f}')
                    runningLoss = 0.0

        print('Finished Training')
        torch.save(network.state_dict(), modelPath)

    # Load saved model for evaluation
    network = Net()
    network.load_state_dict(torch.load(modelPath, weights_only = True))

    # Preview a batch of test images with ground truth and predictions
    testIterator = iter(testLoader)
    images, labels = next(testIterator)

    showImage(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    outputs = network(images)
    _, predicted = torch.max(outputs, 1)  # index of highest score is the predicted class
    print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    # Overall accuracy across all test images
    correctCount = 0
    totalCount = 0
    with torch.no_grad():  # no gradients needed during evaluation
        for batch in testLoader:
            images, labels = batch
            outputs = network(images)
            _, predicted = torch.max(outputs, 1)  # take highest-scoring class as prediction
            totalCount += labels.size(0)
            correctCount += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correctCount // totalCount} %')

    # Per-class accuracy breakdown
    correctPerClass = {className: 0 for className in classes}
    totalPerClass = {className: 0 for className in classes}

    with torch.no_grad():
        for batch in testLoader:
            images, labels = batch
            outputs = network(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correctPerClass[classes[label]] += 1
                totalPerClass[classes[label]] += 1

    for className, classCorrectCount in correctPerClass.items():
        accuracy = 100 * float(classCorrectCount) / totalPerClass[className]
        print(f'Accuracy for class: {className:5s} is {accuracy:.1f} %')
