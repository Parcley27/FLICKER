import sys
from pathlib import Path

import torch
import torch.nn as nn

repoRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repoRoot))

from config import globalBins, localBins, secondaryBins

class ConvolutionTower(nn.Module):
    def __init__(self, channelsIn, inputLength):
        
        super().__init__()

        # layer(in, out, kernel)
        # 16 32 64 is a standard starting point
        # 5 wide to look at 5 adj time steps at once
        self.layers = nn.Sequential(
            # first layer (input)
            nn.Conv1d(channelsIn, 16, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # second (first hidden)
            nn.Conv1d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # third (second hidden)
            nn.Conv1d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # flatten (output layer)
            nn.Flatten(),
        
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channelsIn, inputLength)
            self.outputDimension = self.layers(dummy).numel()
        
    def forward(self, x):
        return self.layers(x)

class TransitClassifier(nn.Module):
    def __init__(self, dropout = 0.5):
        super().__init__()

        self.dropout = dropout

        self.globalTower = ConvolutionTower(3, 201)
        self.localTower = ConvolutionTower(2, 61)
        self.secondaryTower = ConvolutionTower(2, 61)
        
        # 12 scalars plus the three curve views
        fullyConnectedInput = 12 + self.globalTower.outputDimension + self.localTower.outputDimension + self.secondaryTower.outputDimension

        self.fullyConnected = nn.Sequential(
            # input layer
            nn.Linear(fullyConnectedInput, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # hidden layer 1
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # hidden layer 2
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # output layer
            nn.Linear(64, 1),

        )
    
    def forward(self, batch):
        globalView = batch["globalView"]
        localView = batch["localView"]
        secondaryView = batch["secondaryView"]
        scalars = batch["scalars"]

        globalTower = self.globalTower(globalView)
        localTower = self.localTower(localView)
        secondaryTower = self.secondaryTower(secondaryView)

        # combine everything into large vector/tensor thingy
        vector = torch.cat([globalTower, localTower, secondaryTower, scalars], dim = 1)

        tensor = self.fullyConnected(vector)
        # squeeze from (batch, 1) to (batch,) for loss function
        tensor = tensor.squeeze(1)

        return tensor
    
if __name__ == "__main__":
    transitClassifier = TransitClassifier()

    dummyGlobalView = torch.zeros(4, 3, 201)
    dummyLocalView = torch.zeros(4, 2, 61)
    dummySecondaryView = torch.zeros(4, 2, 61)
    dummyScalars = torch.zeros(4, 12)

    dummyBatch = {
        "globalView": dummyGlobalView,
        "localView": dummyLocalView,
        "secondaryView": dummySecondaryView,
        "scalars": dummyScalars,

    }

    tensor = transitClassifier(dummyBatch)
    print(tensor.shape)